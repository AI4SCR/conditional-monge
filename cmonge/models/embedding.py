from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn import manifold
from tqdm import tqdm

from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.metrics import wasserstein_distance
from cmonge.models.rdkit import rdkit_feats


class BaseEmbedding:
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.embeddings = {}


class DoseEmbedding(BaseEmbedding):
    def __init__(self, datamodule: ConditionalDataModule) -> None:
        super().__init__(datamodule.batch_size)
        self.datamodule = datamodule
        embeddings = {
            cond: [int(cond.split("-")[1]) / 10000]
            for cond in self.datamodule.conditions
        }
        self.embeddings = embeddings

    def __call__(self, condition: str):
        condition = self.embeddings[condition]
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch


class RDKitEmbedding(BaseEmbedding):
    def __init__(
        self,
        checkpoint: bool,
        smile_path: str,
        drug_to_smile_path: str,
        name: str,
        model_dir: str,
        datamodule: ConditionalDataModule,
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.smile_path = Path(smile_path)
        self.drug_to_smile_path = Path(drug_to_smile_path)
        self.model_dir = Path(model_dir)
        if not checkpoint:
            logger.info("Calculating RDKit embedding vectors")
            smiles_df = pd.read_csv(self.smile_path)
            smiles_list = smiles_df["smiles"].values

            # caclulate mebeddings
            n_jobs = 16
            data = Parallel(n_jobs=n_jobs)(
                delayed(rdkit_feats)(smiles)
                for smiles in tqdm(smiles_list, position=0, leave=True)
            )

            # clean data, drop nans and infs
            embedding = np.array(data)

            drug_idx, feature_idx = np.where(np.isnan(embedding))
            drug_idx_infs, feature_idx_infs = np.where(np.isinf(embedding))

            drug_idx = np.concatenate((drug_idx, drug_idx_infs))
            feature_idx = np.concatenate((feature_idx, feature_idx_infs))
            embedding[drug_idx, feature_idx] = 0

            # load smile representation of drugs
            drugs = pd.read_csv(self.drug_to_smile_path)
            drugs["drug"] = drugs["drug"].apply(lambda x: x.lower())
            drugs = drugs[
                drugs["drug"].isin(
                    [
                        "abexinostat",
                        "belinostat",
                        "dacinostat",
                        "entinostat",
                        "givinostat",
                        "mocetinostat",
                        "pracinostat",
                        "tacedinaline",
                        "trametinib",
                    ]
                )
            ][["drug", "smile"]]

            df = pd.DataFrame(
                data=embedding,
                index=smiles_list,
                columns=[f"latent_{i}" for i in range(embedding.shape[1])],
            )

            # Drop first feature from generator (RDKit2D_calculated)
            df.drop(columns=["latent_0"], inplace=True)

            # Drop columns with 0 standard deviation
            threshold = 0.01
            columns = [f"latent_{idx+1}" for idx in np.where(df.std() <= threshold)[0]]
            print(f"Deleting columns with std<={threshold}: {columns}")
            df.drop(
                columns=[f"latent_{idx+1}" for idx in np.where(df.std() <= 0.01)[0]],
                inplace=True,
            )

            # normalize and merge with drug names
            normalized_df = (df - df.mean()) / df.std()
            df = drugs.merge(normalized_df, left_on="smile", right_index=True)

            self.model_dir.mkdir(parents=True, exist_ok=True)
            model_dir = self.model_dir / name
            df.to_csv(model_dir)
        else:
            logger.info("Loading RDKit embeddings.")
            model_dir = self.model_dir / name
            df = pd.read_csv(model_dir)

        filter_col = [col for col in df if col.startswith("latent")]
        for index, row in df.iterrows():
            name = row["drug"]
            values = jnp.asarray(row[filter_col].values.astype("float"))
            self.embeddings[name] = values

    def __call__(self, condition: str):
        cond, dose = condition.split("-")
        condition = self.embeddings[cond]
        condition = jnp.append(condition, np.log(int(dose)))
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch


class ModeOfActionEmbedding(BaseEmbedding):
    def __init__(
        self,
        datamodule: ConditionalDataModule,
        checkpoint: bool,
        name: str,
        model_dir: str,
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.model_dir = Path(model_dir)
        if not checkpoint:
            labels = datamodule.train_conditions
            similarity_matrix = jnp.full(
                shape=(len(labels), len(labels)), fill_value=jnp.inf
            )

            cond_to_loaders = datamodule.train_dataloaders()
            with tqdm(total=len(labels) ** 2) as pbar:
                for i, (cond_i, loader_i) in enumerate(cond_to_loaders.items()):
                    for j, (cond_j, loader_j) in enumerate(cond_to_loaders.items()):
                        if (
                            similarity_matrix.at[i, j].get() == jnp.inf
                            and similarity_matrix.at[j, i].get() == jnp.inf
                        ):
                            target_batch_i = next(loader_i[1])
                            target_batch_j = next(loader_j[1])
                            w_dist = wasserstein_distance(
                                target_batch_i, target_batch_j
                            )
                            similarity_matrix = similarity_matrix.at[i, j].set(w_dist)
                            similarity_matrix = similarity_matrix.at[j, i].set(w_dist)
                        pbar.update(1)

            similarity = pd.DataFrame(
                data=similarity_matrix, columns=labels, index=labels
            )
            embedding, stress = manifold.smacof(
                similarity_matrix, metric=True, n_components=10
            )
            embedding_norm = (embedding - embedding.min()) / (
                embedding.max() - embedding.min()
            )
            smacof_10d = pd.DataFrame(data=embedding_norm.T, columns=labels)

            self.model_dir.mkdir(parents=True, exist_ok=True)
            model_dir = self.model_dir / name
            smacof_10d.to_csv(model_dir)
            model_dir = self.model_dir / "similarity"
            similarity.to_csv(model_dir)
        else:
            model_dir = self.model_dir / name
            smacof_10d = pd.read_csv(model_dir)
            smacof_10d = smacof_10d.drop(columns=["Unnamed: 0"])

        for index, row in smacof_10d.T.iterrows():
            values = jnp.asarray(row.values.astype("float"))
            self.embeddings[index] = values

    def __call__(self, condition: str):
        cond, dose = condition.split("-")
        condition = self.embeddings[condition]
        condition = jnp.append(condition, np.log(int(dose)))
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch


class CAR11DimEmbedding(BaseEmbedding):
    def __init__(
        self,
        datamodule: ConditionalDataModule,
        checkpoint: bool,
        name: str,
        model_dir: str,
        batch_size=None,  # For compatability
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.model_dir = Path(model_dir)
        if not checkpoint:
            labels = datamodule.train_conditions

            car_11d = pd.DataFrame([self.encode_CAR_11dim(label) for label in labels]).T
            car_11d.colums = labels
            self.model_dir.mkdir(parents=True, exist_ok=True)
            model_dir = self.model_dir / name
            car_11d.to_csv(model_dir)
        else:
            model_dir = self.model_dir / name
            car_11d = pd.read_csv(model_dir)
            car_11d = car_11d.drop(columns=["Unnamed: 0"])

        for index, row in car_11d.T.iterrows():
            values = jnp.asarray(row.values.astype("float"))
            self.embeddings[index] = values

    def encode_CAR_11dim(self, CAR: str) -> list:
        """
        Compute one-hot encoding of CAR variant on 15 bits.
        Use alphabetical order of CAR domains: 41BB, CD28, CD40, CTLA4, IL15RA.
        For each domain there are 3 bits:
            - Domain present
            - 1st position
            - 2nd position
        So the three bits of CAR with domain A in the first position would be [1,1,0].
        A in second position would be [1,0,1]
        and for the CAR with both domains A [1,1,1].
        The 3 bits for the cars are concatenated into 15 bit. Then the l6th bit is to
        indicate wether CD3z (`z`) is present.
        0 everywhere is TCR-
        """
        all_domains = ["41BB", "CD28", "CD40", "CTLA4", "IL15RA"]
        CAR_variant = CAR.split("-")

        encoding = [0] * 11

        if CAR_variant[0] != "NA":
            # First mark first domain
            index_1 = all_domains.index(CAR_variant[0])
            encoding[index_1] = 1

        # Mark second domain if present
        if CAR_variant[1] != "NA":
            index_2 = all_domains.index(CAR_variant[1])
            encoding[index_2 + 5] = 1

        if CAR_variant[2] == "z":
            encoding[-1] = 1

        return encoding

    def __call__(self, condition: str):
        condition = self.embeddings[condition]
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch


class CAR16DimEmbedding(BaseEmbedding):
    def __init__(
        self,
        datamodule: ConditionalDataModule,
        checkpoint: bool,
        name: str,
        model_dir: str,
        batch_size=None,  # For compatability
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.model_dir = Path(model_dir)
        if not checkpoint:
            labels = datamodule.train_conditions

            car_16d = pd.DataFrame([self.encode_CAR_16dim(label) for label in labels]).T
            car_16d.colums = labels
            self.model_dir.mkdir(parents=True, exist_ok=True)
            model_dir = self.model_dir / name
            car_16d.to_csv(model_dir)
        else:
            model_dir = self.model_dir / name
            car_16d = pd.read_csv(model_dir)
            car_16d = car_16d.drop(columns=["Unnamed: 0"])

        for index, row in car_16d.T.iterrows():
            values = jnp.asarray(row.values.astype("float"))
            self.embeddings[index] = values

    def encode_CAR_16dim(CAR):
        """
        Compute one-hot encoding of CAR variant on 16 bits.
        Use alphabetical order of CAR domains: 41BB, CD28, CD40, CTLA4, IL15RA.
        For each domain there are 3 bits:
            - Domain present
            - 1st position
            - 2nd position
        So the three bits of CAR with domain A in the first position would be [1,1,0].
        A in second position would be [1,0,1]
        and for the CAR with both domains A [1,1,1].
        The 3 bits for the cars are concatenated into 15 bit. Then the 16th bit is to
        indicate wether CD3z (`z`) is present.
        0 everywhere is TCR-
        """
        all_domains = ["41BB", "CD28", "CD40", "CTLA4", "IL15RA"]
        CAR_variant = CAR.split("-")

        encoding = [0] * 16

        if CAR_variant[0] != "NA":
            # First mark first domain
            index_1 = all_domains.index(CAR_variant[0])
            encoding[index_1 * 3] = 1
            encoding[index_1 * 3 + 1] = 1

        # Mark second domain if present
        if CAR_variant[1] != "NA":
            index_2 = all_domains.index(CAR_variant[1])
            encoding[index_2 * 3] = 1
            encoding[index_2 * 3 + 2] = 1

        if CAR_variant[2] == "z":
            encoding[-1] = 1

        return encoding

    def __call__(self, condition: str):
        condition = self.embeddings[condition]
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch


class CarEsmSmall(BaseEmbedding):
    def __init__(
        self,
        datamodule: ConditionalDataModule,
        checkpoint: bool,
        name: str,
        model_dir: str,
        batch_size=None,  # For compatability
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.model_dir = Path(model_dir)
        if not checkpoint:
            logger.error(
                """ESM embedding only works with checkpoint,
                please save a pre-computed embedding"""
            )
        else:
            model_dir = self.model_dir / name
            embed = pd.read_csv(model_dir)
            embed = embed.drop(columns=["Unnamed: 0"])

        for index, row in embed.T.iterrows():
            values = jnp.asarray(row.values.astype("float"))
            self.embeddings[index] = values

    def __call__(self, condition: str):
        condition = self.embeddings[condition]
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch


class MetaDataEmbedding(BaseEmbedding):
    def __init__(
        self,
        datamodule: ConditionalDataModule,
        checkpoint: bool,
        name: str,
        model_dir: str,
        batch_size=None,  # For compatability
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.model_dir = Path(model_dir)
        dataset = datamodule.data_config.file_path.split("/")[-1][:-5]
        if not checkpoint:
            adata = datamodule.data_config.parent
            group = "CAR_Variant"
            cont_scores = [
                "Cytotoxicity_1",
                "Proinflamatory_2",
                "Memory_3",
                "CD4_Th1_4",
                "CD4_Th2_5",
                "S.Score",
                "G2M.Score",
            ]
            fraction_scores = ["Donor", "Time", "Phase", "ident", "subset"]

            means = adata.obs[[group] + cont_scores].groupby(group).mean()
            stds = adata.obs[[group] + cont_scores].groupby(group).std()
            cont_features = means.merge(
                stds, left_index=True, right_index=True, suffixes=("_mean", "_std")
            )

            group_size = adata.obs.groupby(group).size()
            all_cat_counts = []
            for cat in fraction_scores:
                temp = (
                    adata.obs.groupby([group, cat], observed=False)
                    .size()
                    .reset_index(drop=False)
                )
                temp = pd.pivot_table(
                    data=temp, index=group, columns=cat, values=0, observed=False
                )
                all_cat_counts.append(temp)

            cat_features = pd.concat(all_cat_counts, axis=1)
            cat_features = cat_features.div(group_size, axis=0)

            embedding = cont_features.merge(
                cat_features, left_index=True, right_index=True
            ).T

            self.model_dir.mkdir(parents=True, exist_ok=True)
            model_dir = self.model_dir / f"{dataset}_{name}"
            embedding.to_csv(model_dir)
        else:
            model_dir = self.model_dir / f"{dataset}_{name}"
            embedding = pd.read_csv(model_dir)
            embedding = embedding.drop(columns=["Unnamed: 0"])

        for index, row in embedding.T.iterrows():
            values = jnp.asarray(row.values.astype("float"))
            self.embeddings[index] = values

    def __call__(self, condition: str):
        condition = self.embeddings[condition]
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch


embed_factory = {
    "rdkit": RDKitEmbedding,
    "dose": DoseEmbedding,
    "moa": ModeOfActionEmbedding,
    "embed_11d": CAR11DimEmbedding,
    "embed_16d": CAR16DimEmbedding,
    "esm_small": CarEsmSmall,
    "esm_small_full_dim": CarEsmSmall,
    "esm_small_full_seq": CarEsmSmall,
    "esm_small_tail_dim": CarEsmSmall,
    "esm_small_tail_seq": CarEsmSmall,
    "metadata": MetaDataEmbedding,
}

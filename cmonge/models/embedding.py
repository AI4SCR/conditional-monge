from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.metrics import wasserstein_distance
from cmonge.models.rdkit import rdkit_feats
from joblib import Parallel, delayed
from loguru import logger
from sklearn import manifold
from tqdm import tqdm


class BaseEmbedding:
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.embeddings = {}


class DoseEmbedding(BaseEmbedding):
    def __init__(self, datamodule: ConditionalDataModule) -> None:
        super().__init__(datamodule.batch_size)
        self.datamodule = datamodule
        embeddings = {cond: [int(cond.split("-")[1]) / 10000] for cond in self.datamodule.conditions}
        self.embeddings = embeddings

    def __call__(self, condition: str):
        condition = self.embeddings[condition]
        condition_batch = jnp.asarray([condition for _ in range(self.batch_size)])
        return condition_batch


class RDKitEmbedding(BaseEmbedding):
    def __init__(
        self,
        batch_size: int,
        checkpoint: bool,
        smile_path: str,
        drug_to_smile_path: str,
        name: str,
        model_dir: str,
        datamodule=None,# For compatability
    ) -> None:
        super().__init__(batch_size)
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
                delayed(rdkit_feats)(smiles) for smiles in tqdm(smiles_list, position=0, leave=True)
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
                data=embedding, index=smiles_list, columns=[f"latent_{i}" for i in range(embedding.shape[1])]
            )

            # Drop first feature from generator (RDKit2D_calculated)
            df.drop(columns=["latent_0"], inplace=True)

            # Drop columns with 0 standard deviation
            threshold = 0.01
            columns = [f"latent_{idx+1}" for idx in np.where(df.std() <= threshold)[0]]
            print(f"Deleting columns with std<={threshold}: {columns}")
            df.drop(columns=[f"latent_{idx+1}" for idx in np.where(df.std() <= 0.01)[0]], inplace=True)

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
        batch_size=None, # For compatability
    ) -> None:
        super().__init__(datamodule.batch_size)
        self.model_dir = Path(model_dir)
        if not checkpoint:
            labels = datamodule.train_conditions
            similarity_matrix = jnp.full(shape=(len(labels), len(labels)), fill_value=jnp.inf)

            cond_to_loaders = datamodule.train_dataloaders()
            with tqdm(total=len(labels) ** 2) as pbar:
                for i, (cond_i, loader_i) in enumerate(cond_to_loaders.items()):
                    for j, (cond_j, loader_j) in enumerate(cond_to_loaders.items()):
                        if similarity_matrix.at[i, j].get() == jnp.inf and similarity_matrix.at[j, i].get() == jnp.inf:
                            target_batch_i = next(loader_i[1])
                            target_batch_j = next(loader_j[1])
                            w_dist = wasserstein_distance(target_batch_i, target_batch_j)
                            similarity_matrix = similarity_matrix.at[i, j].set(w_dist)
                            similarity_matrix = similarity_matrix.at[j, i].set(w_dist)
                        pbar.update(1)

            similarity = pd.DataFrame(data=similarity_matrix, columns=labels, index=labels)
            embedding, stress = manifold.smacof(similarity_matrix, metric=True, n_components=10)
            embedding_norm = (embedding - embedding.min()) / (embedding.max() - embedding.min())
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


embed_factory = {"rdkit": RDKitEmbedding, "dose": DoseEmbedding, "moa": ModeOfActionEmbedding}

from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.datasets.single_loader import SciPlexModule


class TestConditionalDataModule:
    def test_homogeneous_splitter(self, synthetic_config):
        config = synthetic_config.copy()
        config.condition.mode = "homogeneous"
        config.condition.conditions = ["givinostat-10", "givinostat-100"]
        config.condition.split = [0.6, 0.2, 0.2]
        module = ConditionalDataModule(config.data, config.condition, config.ae)

        module.splitter()

        assert len(module.loaders) == 2
        assert isinstance(module.loaders["givinostat-10"], SciPlexModule)

        assert len(module.loaders["givinostat-10"].control_train_cells) == 55
        assert len(module.loaders["givinostat-10"].control_valid_cells) == 18
        assert len(module.loaders["givinostat-10"].control_test_cells) == 19

    def test_extrapolate_splitter(self, synthetic_config):
        config = synthetic_config.copy()
        config.condition.mode = "extrapolate"
        config.condition.conditions = ["givinostat-10", "givinostat-100"]
        config.condition.split = [0.8, 0.2]
        config.condition.ood = ["givinostat-100"]
        config.condition.ood_split = [0, 1]
        module = ConditionalDataModule(config.data, config.condition, config.ae)

        assert len(module.loaders) == 2
        assert isinstance(module.loaders["givinostat-100"], SciPlexModule)
        train_condition = module.train_conditions[0]
        assert len(module.loaders[train_condition].control_train_cells) == 73
        assert len(module.loaders[train_condition].control_valid_cells) == 19
        assert len(module.loaders[train_condition].control_test_cells) == 0

        valid_condition = module.valid_conditions[0]
        assert len(module.loaders[valid_condition].control_train_cells) == 73
        assert len(module.loaders[valid_condition].control_valid_cells) == 19
        assert len(module.loaders[valid_condition].control_test_cells) == 0

        assert len(module.train_conditions) == 1
        assert len(module.valid_conditions) == 2

    def test_sample_condition(self, synthetic_config):
        config = synthetic_config.copy()
        config.condition.mode = "extrapolate"
        config.condition.conditions = ["givinostat-10", "givinostat-100"]
        config.condition.split = [1, 0]
        config.condition.ood = ["givinostat-100"]
        module = ConditionalDataModule(config.data, config.condition, config.ae)

        train_cond = module.sample_condition("train")
        valid_cond = module.sample_condition("valid")

        assert train_cond in "givinostat-10"
        assert valid_cond == "givinostat-100"

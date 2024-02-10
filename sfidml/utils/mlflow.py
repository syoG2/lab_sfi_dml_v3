import tempfile
from pathlib import Path
from shutil import rmtree
from typing import List

import mlflow


class OutputMLFlow:
    def __init__(self, tracking_uri: str, experiment_name: str, run_name: str) -> None:
        self.tracking_uri = Path(tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.client = mlflow.tracking.MlflowClient()

        (self.tracking_uri / ".trash").mkdir(parents=True, exist_ok=True)

        self._create_experiment()
        self._trash_incomplete_run()
        self._trash_duplicate_run()

    def _create_experiment(self) -> None:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(name=self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        (self.tracking_uri / ".trash" / self.experiment_id).mkdir(
            parents=True, exist_ok=True
        )

    def _trash_incomplete_run(self) -> None:
        for path in (self.tracking_uri / self.experiment_id).iterdir():
            if not path.is_file():
                for item in ["meta.yaml", "artifacts", "metrics", "params", "tags"]:
                    if not (path / item).exists():
                        rmtree(path)
                        break

    def _trash_duplicate_run(self) -> None:
        run_id = None
        for run in self.client.search_runs(self.experiment_id):
            if run.data.tags["mlflow.runName"] == self.run_name:
                run_id = run.info.run_id
            if run_id is not None:
                (self.tracking_uri / self.experiment_id / run_id).replace(
                    (self.tracking_uri / ".trash" / self.experiment_id / run_id)
                )
                run_id = None


class InputMLFlow:
    def __init__(self, tracking_uri: str, experiment_name: str, run_name: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.client = mlflow.tracking.MlflowClient()

        self._get_run_id()

    def _get_run_id(self) -> None:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        for run in self.client.search_runs(self.experiment_id):
            if "mlflow.runName" in run.data.tags:
                if run.data.tags["mlflow.runName"] != self.run_name:
                    continue
            self.params = run.data.params
            self.run_id = run.info.run_id


class SearchCandidatesMLFlow:
    def __init__(
        self, tracking_uri: str, experiment_name: str, searched_key_list: List[str]
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name
        self.searched_key_list = searched_key_list
        self.client = mlflow.tracking.MlflowClient()

        self._get_run_id()

    def _get_run_id(self) -> None:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        self.params_list, self.run_id_list, self.metrics_list = [], [], []
        for run in self.client.search_runs(self.experiment_id):
            if "mlflow.runName" in run.data.tags:
                run_name_key_list = run.data.tags["mlflow.runName"].split("-")
                matched_key_list = set(run_name_key_list) & set(self.searched_key_list)
                if len(matched_key_list) == len(self.searched_key_list):
                    self.params_list.append(run.data.params)
                    self.run_id_list.append(run.info.run_id)
                    self.metrics_list.append(run.data.metrics)


class GetScoreMLFlow:
    def __init__(self, tracking_uri: str, experiment: str, split: int) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

        self.experiment = experiment
        self.split = split

        self._scoring()

    def _scoring(self) -> None:
        self.scores = {}
        for i in range(self.split):
            experiment_name = self.experiment + "_" + str(i)
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

            for run in self.client.search_runs(experiment_id):
                run_name = run.data.tags["mlflow.runName"]
                if run_name not in self.scores:
                    self.scores[run_name] = {experiment_name: dict(run.data.metrics)}
                else:
                    self.scores[run_name][experiment_name] = dict(run.data.metrics)


def output_mlflow_df(file_name, df):
    with tempfile.TemporaryDirectory() as t:
        artifact_path = Path(t) / file_name
        df.to_json(
            artifact_path,
            orient="records",
            force_ascii=False,
            lines=True,
        )
        mlflow.log_artifact(artifact_path)

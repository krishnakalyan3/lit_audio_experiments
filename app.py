#!/usr/bin/env python

import os
import lightning as L
from typing import Optional
import subprocess
from lightning_app.storage import Drive
from lit_vscode import VSCodeServer

class MLFlowWork(L.LightningWork):
    def __init__(self, cloud_compute: Optional[L.CloudCompute] = None):
        super().__init__(cloud_compute=cloud_compute)
        self.ml_ulr = None
        self.storage_path = None

    def run(self, drive: Drive):
        mlflow_root = os.path.join(drive.root, "mlruns")
        print(mlflow_root)

        drive.get("mlruns/")
        #subprocess.run(f"aws s3 s3://{mlflow_root}", shell=True, env=os.environ.copy())

        cmd1 = f"ls -ltr ."
        subprocess.run(cmd1, shell=True, env=os.environ.copy())

        cmd1 = f"mlflow ui  -h {self.host} -p {self.port} --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root mlruns"
        subprocess.run(cmd1, shell=True, env=os.environ.copy())


class SVMWork(L.LightningWork):
    
    def __init__(self, cloud_compute: Optional[L.CloudCompute] = None):
        super().__init__(cloud_compute=cloud_compute)
        self.storage = None

    def run(self, drive: Drive):
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        import mlflow
        import mlflow.sklearn

        X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        lr = LogisticRegression()
        lr.fit(X, y)
        score = lr.score(X, y)
        print("Score: %s" % score)
        mlflow.log_metric("score", score)
        mlflow.sklearn.log_model(lr, "model")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
        print("Traning Completed")
        drive.put("mlruns")


class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.drive_1 = Drive("lit://drive_1")
        self.mlflow_work = MLFlowWork(cloud_compute=L.CloudCompute(os.getenv("COMPUTE", "cpu-small")))
        self.ml_pipeline = SVMWork(cloud_compute=L.CloudCompute(os.getenv("COMPUTE", "cpu-small")))
        #self.vscode_work = VSCodeServer(cloud_compute=L.CloudCompute(os.getenv("COMPUTE", "cpu-small")))

    def run(self):
        #self.vscode_work.run()
        self.ml_pipeline.run(self.drive_1)
        self.mlflow_work.run(self.drive_1)
    
    def configure_layout(self):
        tab_1 = {
            "name": "MLFlow",
            "content": self.mlflow_work
        }
        #tab_2 = {
        #    "name": "VSCode",
        #    "content": self.vscode_work
        #}

        return [tab_1]

app = L.LightningApp(RootFlow())
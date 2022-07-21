#!/usr/bin/env python

import os
import lightning as L
from typing import Optional
import subprocess
from lightning_app.storage import Drive
#from lit_vscode import VSCodeServer

class MLFlowWork(L.LightningWork):
    def __init__(self, cloud_compute: Optional[L.CloudCompute] = None):
        super().__init__(cloud_compute=cloud_compute, parallel=True)
        self.ml_ulr = None
        self.drive_1 = "mlflow_artifact"

    def run(self):
        os.mkdir(self.drive_1)
        cmd1 = f"mlflow ui  -h {self.host} -p {self.port} --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root {self.drive_1}"
        subprocess.run(cmd1, shell=True)


class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.mlflow_work = MLFlowWork(cloud_compute=L.CloudCompute(os.getenv("COMPUTE", "cpu-small")))
        #self.vscode_work = VSCodeServer(cloud_compute=L.CloudCompute(os.getenv("COMPUTE", "cpu-small")))

    def run(self):
        self.mlflow_work.run()
    
    def configure_layout(self):
        return [{'name': "MLFlow", 'content': self.mlflow_work}]

app = L.LightningApp(RootFlow())
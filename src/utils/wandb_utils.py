import wandb
import os
import torch

class WandbLogger:
    def __init__(
        self,
        config,
        output_dir=None,
        job_type="training",
    ):
        self.config = config
        self.entity = config["entity"]
        self.project_name = config["project_name"]
        self.run_name = config["run_name"]
        self.output_dir = output_dir
        self.initialized = False
        self.job_type = job_type

    def initialize(self):
        if not self.initialized:
            self.run = wandb.init(
                entity=self.entity,
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                dir=self.output_dir,
                job_type=self.job_type,
            )
            self.initialized = True

    def log(self, data):
        if not self.initialized:
            initialize()
        wandb.log(data)

    def save_model(self, model, model_name, output_dir):
        if not self.initialized:
            initialize()
        file_path = os.path.join(output_dir, model_name)
        torch.save(model.state_dict(), file_path)
        artifact = wandb.Artifact('model_weights', type='model')
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

    def finish(self):
        if self.initialized:
            wandb.finish()
            self.initialized = False

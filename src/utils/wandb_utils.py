import wandb
import os

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
            wandb.init(
                entity=self.entity,
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                dir=self.output_dir,
                job_type=self.job_type,
            )
            self.initialized = True

    def log(self, data):
        if self.initialized:
            wandb.log(data)

    def save_model(self, output_dir):
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(os.path.join(output_dir, "wandb_model_checkpoint.pt"))
        wandb.log_artifact(artifact)
        logging.info("Model checkpoint logged to wandb.")

    def finish(self):
        if self.initialized:
            wandb.finish()
            self.initialized = False

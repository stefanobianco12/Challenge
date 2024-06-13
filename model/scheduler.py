from apscheduler.schedulers.blocking import BlockingScheduler
from train import Trainer

print("START SCHEDULER")
trainer=Trainer(threshold=0.8)
scheduler =BlockingScheduler()
scheduler.add_job(trainer.training, 'interval', minutes=5)
scheduler.start()



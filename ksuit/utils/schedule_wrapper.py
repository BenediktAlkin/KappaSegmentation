from kappaschedules import ScheduleBase, ConstantSchedule

from ksuit.utils.update_counter import UpdateCounter


class ScheduleWrapper:
    def __init__(self, schedule: ScheduleBase, update_counter: UpdateCounter = None, interval="update"):
        self.schedule = schedule
        self.update_counter = update_counter
        self.interval = interval

    def get_value(self):
        if self.update_counter is None:
            assert isinstance(self.schedule, ConstantSchedule)
            return self.schedule.get_value(step=0, total_steps=1)
        if self.interval == "update":
            return self.schedule.get_value(
                step=self.update_counter.cur_checkpoint.update,
                total_steps=self.update_counter.end_checkpoint.update,
            )
        elif self.interval == "epoch":
            assert isinstance(self.update_counter.cur_checkpoint.epoch, int)
            return self.schedule.get_value(
                step=self.update_counter.cur_checkpoint.epoch * self.update_counter.updates_per_epoch,
                total_steps=self.update_counter.end_checkpoint.update,
            )
        else:
            raise NotImplementedError(f"invalid interval: {self.interval}")

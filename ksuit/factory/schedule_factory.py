from kappaschedules import object_to_schedule, ConstantSchedule

from ksuit.utils.schedule_wrapper import ScheduleWrapper
from .base import FactoryBase


class ScheduleFactory(FactoryBase):
    def create(self, obj_or_kwargs, collate_fn=None, **kwargs):
        assert collate_fn is None
        interval = "update"
        if isinstance(obj_or_kwargs, (float, int)):
            schedule = ConstantSchedule(value=obj_or_kwargs)
            update_counter = None
        else:
            update_counter = kwargs.pop("update_counter", None)
            # check interval of schedules defined via dict (e.g. linear_warmup_cosine_decay_schedule)
            if isinstance(obj_or_kwargs, dict) and "interval" in obj_or_kwargs:
                interval = obj_or_kwargs.pop("interval")
            # check interval of schedules defined via list (e.g. [linear_increasing, cosine_decay])
            elif isinstance(obj_or_kwargs, list):
                # expect interval in either all schedules, or none; mixed is not supported
                if isinstance(obj_or_kwargs[0], dict) and "interval" in obj_or_kwargs[0]:
                    expected_interval = obj_or_kwargs[0]["interval"]
                else:
                    expected_interval = None
                if expected_interval is not None:
                    for item in obj_or_kwargs:
                        assert item.pop("interval") == expected_interval, "use interval in all schedules or none"
                    interval = expected_interval
            if update_counter is not None:
                assert "batch_size" not in kwargs
                assert "updates_per_epoch" not in kwargs
                kwargs["batch_size"] = update_counter.effective_batch_size
                kwargs["updates_per_epoch"] = update_counter.updates_per_epoch
            schedule = object_to_schedule(obj=obj_or_kwargs, **kwargs)
        if schedule is None:
            return None
        return ScheduleWrapper(schedule=schedule, update_counter=update_counter, interval=interval)

    def instantiate(self, kind, optional_kwargs=None, **kwargs):
        raise RuntimeError

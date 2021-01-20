from fastai.basics import Callback


class AzureRunLogCallback(Callback):
    "Log losses and metrics to Azure"
    def __init__(self, run_context):
        self.run_context = run_context

    def after_epoch(self):
        # log metrics
        for n, v in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if n not in ['epoch', 'time']:
                if isinstance(v, dict):
                    for km, vm in v.items():
                        self.run_context.log(f'{n}_{km}', vm)
                else:
                    self.run_context.log(f'{n}', v)
        return True

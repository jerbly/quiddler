from icevision.all import *

class AzureRunLogCallback(fastai.Callback):
    "Log losses and metrics to Azure"
    def __init__(self, run_context):
        self.run_context = run_context

    def after_epoch(self):
        # log metrics
        for n, v in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if n not in ['epoch', 'time']:
                if isinstance(v,dict):
                    for km, vm in v.items():
                        self.run_context.log(f'{n}_{km}', vm)
                else:
                    self.run_context.log(f'{n}', v)
        return True

class ViaParser(parsers.FasterRCNN, parsers.FilepathMixin, parsers.SizeMixin):
    def __init__(self, js, source, class_map):
        self.js = js
        self.source = source
        self.class_map = class_map

    def __iter__(self):
        yield from self.js.items()

    def __len__(self):
        return len(self.js.items())

    def imageid(self, o) -> Hashable:
        return o[1]['filename']

    def filepath(self, o) -> Union[str, Path]:
        return self.source / f"{o[1]['filename']}"

    def height(self, o) -> int:
        return open_img(self.source/o[1]['filename']).shape[0]

    def width(self, o) -> int:
        return open_img(self.source/o[1]['filename']).shape[1]

    def labels(self, o) -> List[int]:
        labels = []
        for shape in o[1]['regions']:
            label = shape['region_attributes']['label']
            if label in self.class_map.class2id:
                labels.append(self.class_map.get_name(label))
        return labels

    def bboxes(self, o) -> List[BBox]:
        boxes = []
        for shape in o[1]['regions']:
            label = shape['region_attributes']['label']
            if label in self.class_map.class2id:    
                x,y = shape['shape_attributes']['all_points_x'],shape['shape_attributes']['all_points_y']
                boxes.append(BBox.from_xyxy(min(x),min(y),max(x),max(y)))
        return boxes

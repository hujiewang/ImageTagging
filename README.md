# Image Classification using VGGNet

## Usage

### Dataset is prepared according to (http://dp.readthedocs.org/en/latest/data/#imagenet)
```
th downloadimagenet.lua
th harmonizeimagenet.lua
th train.lua --batchSize 16 --momentum 0.9
```

* Warning:

  360GB of space required

  Each input(a patch of image) takes 186 MB of memory, be careful about the batch size

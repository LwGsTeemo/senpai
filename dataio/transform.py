from monai.transforms import (
    Compose,
    RandRotate90d, 
    RandFlipd
)

train_transforms = Compose([
    # Orientationd(keys=["image", "label"], axcodes="RAS"),
    # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=[0, 2]),
    RandFlipd(keys=['image', 'label'], prob=0.5),
])

train_transforms.set_random_state(seed=42)

## Processing the dataset

Run the following commands to preprocess the handwritten formula photos:

```bash
python saving.py ../../data/img-to-latex/HME100K/test/test-images ../../data/img-to-latex/preprocessed_HME/test
python saving.py ../../data/img-to-latex/HME100K/train/train-images ../../data/img-to-latex/preprocessed_HME/train
```

To generate sample reviews:
```bash
cd ../scripts
python sample_images.py
```

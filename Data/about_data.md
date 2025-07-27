# DeepFashion Dataset

The dataset located [**here**](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc?resourcekey=0-NWldFxSChFuCpK4nzAIGsg)

the **DeepFashion** repository contain 5 folders, for our purposes we will use just 2

- ***Category and Attribute Prediction Benchmark***

- `images`: our main images, the pixels we'll feed into our classifier / try-on network
- `list_category_cloth`: it contains the names of each clothing category and the integer ID that the dataset uses internally.
for example whenever our code sees a category ID (e.g. 3), you look up its human-readable name (`Dress`) here.
- `list_category_img`: it contains a mapping for each image to the category ID it belongs to.
- `list_attr_cloth`: Defines the universe of possible attributes and assigns each one a numeric ID. ex, When you see an attribute ID (say 7), look here to know that it corresponds to “v-neck” (or whatever the name is).
- `list_attr_img`: For every image, lists a vector indicating which of the 100 attributes are present, absent, or unknown.
- `list_bbox`: Bounding boxes around the garment in each image.
- `list_landmarks` Keypoints on collars, hems, sleeves, etc., for fine-grained alignment.
- `list_eval_partition`: Predefined train/val/test splits (70/15/15) for the DeepFashion benchmark.

# Fashoion-gen Dataset

we download this dataset [**here**](https://drive.usercontent.google.com/download?id=1_1djA1qQphB3dd4b5jTIuoWd4JRV9zVQ&export=download&authuser=0)
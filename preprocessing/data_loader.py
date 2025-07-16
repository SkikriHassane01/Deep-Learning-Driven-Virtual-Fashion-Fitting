#________________TODO: Imports_____________________________________________________________
import pandas as pd
from pathlib import Path
import preprocessing.config as C

#_______________TODO:Load the Deep Fashion Dataset annotation_____________________________
def load_deepFashion()->pd.DataFrame:
    """
    This Function will load metadata from the DeepFashion dataset and return a clean
    dataframe version with this columns:
        - file: where the image located, the file path
        - category_label: we will get it from the `list_categroy_img.txt` file
        - category_name: eg., Blouse, Pants.
        - category_type: the type 1, 2 or 3
        - region: 1: upper, 2:lower, 3: full
        - x_1,  y_1,  x_2,  y_2: coordinate of the boundring box
            [x1,y1]: upper left coordinate of the bbox
            [x2,y2]: lower right coordinate of the bbox
        - split: either "train" or "val" or "test"
    """
    
    # 1. load the list of categorical images 
    ## file <> category_label
    category_images = pd.read_csv(
        C.ANNOTATIONS_DIR / "category" /"list_category_img.txt",
        sep = r"\s+",
        header= None,
        skiprows= 2,
        names= ['file', 'category_label']
    )
    
    # 2. load the list of categorical cloth
    ## category_name <> category_type(1,2,3)
    category_cloth = pd.read_csv(
        C.ANNOTATIONS_DIR / "category" / "list_category_cloth.txt",
        sep = r"\s+",
        skiprows= 2,
        header= None,
        names= ['category_name', 'category_type']
    ).reset_index(names="category_label") #the category_label is the index of the category_cloth now

    # add the region column
    category_cloth["region"] = category_cloth["category_type"].map({1: "upper", 2: "lower", 3: "full"})

    # 3. merge the two dataset 
    df = category_images.merge(category_cloth, on="category_label", how="left")

    # 4. add the bbox coordinate
    bbox_coordinate = pd.read_csv(
        C.ANNOTATIONS_DIR / "bbox" / "list_bbox.txt",
        sep=r"\s+",
        header=None,
        skiprows=2,
        names = ["file", "x_1", "y_1", "x_2", "y_2"]
    )
    
    df = df.merge(bbox_coordinate,on="file" ,how="left")
    
    # 5. add the split info
    split = pd.read_csv(
        C.ANNOTATIONS_DIR  / "splits" / "list_eval_partition.txt",
        sep=r"\s+", header=None, skiprows=2,
        names=["file", "split"]
    )
    df = df.merge(split, on="file", how="left")
    return df

#______________TODO: Load the Fashion-Gen dataset style____________________________________

def load_fashion_ges() -> pd.DataFrame:
    styles = pd.read_csv(C.STYLES_GEN_DIR, on_bad_lines='warn', engine="python")
    
    styles = styles.rename(columns={"id":"file", "subCategory" : "category"})
    
    styles["file"] = styles["file"].astype(str) + ".jpg"
    
    styles["split"] = (
    styles.groupby("category").cumcount()
    .apply(lambda i : "train" if i%10 <8 else ("val" if i%10 == 8 else "test"))
    )
    return styles
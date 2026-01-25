import pandas as pd


def main():
    df = pd.read_csv("data/train.csv")
    
    print("=" * 50)
    print("[Dataset Overview]")
    print(f"Total samples : {len(df)}")
    print(f"Columns       : {list(df.columns)}")
    
    print("\n[Class Distribution]")
    class_counts = df["label"].value_counts().sort_index()
    print(class_counts)
    
    print("\n[Summary]")
    print(f"Number of classes : {class_counts.shape[0]}")
    print(f"Class list        : {class_counts.index.tolist()}")
    print("=" * 50)

if __name__ == "__main__":
    main()


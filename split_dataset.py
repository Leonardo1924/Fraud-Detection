import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_path, train_output_path, test_output_path, target_column, test_size=0.3, random_state=666):
    """
    Splits the dataset into training and testing subsets and saves them as separate files.

    Parameters:
        input_path (str): Path to the input dataset (raw dataset).
        train_output_path (str): Path to save the training dataset.
        test_output_path (str): Path to save the testing dataset.
        target_column (str): Name of the target column for stratified splitting.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.

    Returns:
        None
    """

    # Load the dataset
    data = pd.read_csv(input_path)

    # Split the dataset into training and testing subsets
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        stratify=data[target_column],
        random_state=random_state
    )

    #Save the training and testing datasets
    train_data.to_csv(train_output_path, index=False)
    test_data.to_csv(test_output_path, index=False)

    print("Training and testing datasets saved successfully.")
    print(f"Training dataset saved to {train_output_path}")
    print(f"Testing dataset saved to {test_output_path}")

# Example Usage
if __name__ == "__main__":
    input_path = 'Data/data.csv'
    train_output_path = 'Data/train_data.csv'
    test_output_path = 'Data/test_data.csv'
    target_column = 'is_fraud'
    split_dataset(input_path, train_output_path, test_output_path, target_column)
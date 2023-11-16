import pandas as pd

def load_and_filter_data(file_path, cab_type):
    """
    Load the ride data from the specified file and filter it based on the cab type.

    Parameters:
    file_path (str): The path to the CSV file containing the ride data.
    cab_type (str): The type of cab service to filter for (e.g., 'Lyft').

    Returns:
    pd.DataFrame: The filtered DataFrame containing only the specified cab type's data.
    """

    # Load the dataset from the CSV file
    df = pd.read_csv(file_path)

    # # Filter the DataFrame for the specified cab type
    # filtered_df = df[df['cab_type'] == cab_type]

    # # Prepare the 'product_id' and 'source' columns for easier selection in Streamlit
    # if 'product_id' in filtered_df.columns:
    #     filtered_df['product_id'] = filtered_df['product_id'].astype('category')

    # if 'source' in filtered_df.columns:
    #     filtered_df['source'] = filtered_df['source'].astype('category')

    # 데이터를 필터링하고 카테고리 타입으로 변경
    filtered_df = df[df['cab_type'] == cab_type].copy()  # 복사본을 명시적으로 생성
    filtered_df.loc[:, 'product_id'] = filtered_df['product_id'].astype('category')
    filtered_df.loc[:, 'source'] = filtered_df['source'].astype('category')


    return filtered_df

import tensorflow as tf
import pandas as pd

epochs = 20
batch_size = 32

def main():
    print('Home Credit Default!')

    # train_dataset = tf.contrib.data.make_csv_dataset(
    #         '../credit-risk-torch4/input/final_data_with_feature_engineered.csv',
    #         batch_size

    df = pd.read_csv('./final_data_with_feature_engineered.csv')

    # remove whitespace from pandas names
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace(':', '-')
    df.columns = df.columns.str.replace('(', '_')
    df.columns = df.columns.str.replace(')', '_')
    df.columns = df.columns.str.replace('+', '.')
    df.columns = df.columns.str.replace(',', '_')

    train_df = df[df['TARGET'].notnull()]

    train_x_df = train_df.drop(columns = ['TARGET', 'SK_ID_CURR', 'index'])
    train_y_df = train_df['TARGET']

    train_ds = tf.estimator.inputs.pandas_input_fn(train_x_df, train_y_df, batch_size, epochs, shuffle=True)

    x, y = train_ds()
    feature_cols = [tf.feature_column.numeric_column(k) for k in x]

    tf.logging.set_verbosity(tf.logging.INFO)

    # print('cols', feature_cols)
    config = tf.estimator.RunConfig(log_step_count_steps=1)
    model = tf.estimator.LinearClassifier(feature_columns=feature_cols, config=config)
    model.train(train_ds, steps=100)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(528, kernel_initializer=tf.contrib.layers.xavier_initializer(), use_bias=False)
    # ])

    # loss = tf.keras.backend.binary_crossentropy(
    #     target,
    #     output,
    #     from_logits=True
    # )


    print('done training')






if __name__ == '__main__':
    main()

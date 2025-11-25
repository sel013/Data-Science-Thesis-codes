import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
import optuna
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from carbontracker.tracker import CarbonTracker
import shap
import random

#setting a random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#Loading the data and doing preprocssing

def load_and_preprocess_data(dataset_path='Hourly_Elec_Additional_Data.pkl'):
    df = pd.read_pickle(dataset_path)

    datetime_col = df['datetime_hour']
    insee_col = df['insee_region']
    weather_col = df['weather_code']

    #selecting the numeric columns for minmaxscaler
    numeric_cols = df.drop(columns=['datetime_hour', 'insee_region', 'weather_code']).select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].copy()

    # creating the cyclical features for the time variables, both sin and cos
    df_numeric['sin_hour'] = np.sin(2 * np.pi * df_numeric['hour'] / 24)
    df_numeric['cos_hour'] = np.cos(2 * np.pi * df_numeric['hour'] / 24)
    df_numeric['sin_day'] = np.sin(2 * np.pi * df_numeric['day_of_week'] / 7)
    df_numeric['cos_day'] = np.cos(2 * np.pi * df_numeric['day_of_week'] / 7)
    df_numeric['sin_month'] = np.sin(2 * np.pi * df_numeric['month'] / 12)
    df_numeric['cos_month'] = np.cos(2 * np.pi * df_numeric['month'] / 12)
    df_numeric['sin_quarter'] = np.sin(2 * np.pi * df_numeric['quarter'] / 4)
    df_numeric['cos_quarter'] = np.cos(2 * np.pi * df_numeric['quarter'] / 4)
    df_numeric['sin_week'] = np.sin(2 * np.pi * df_numeric['week_of_year'] / 52)
    df_numeric['cos_week'] = np.cos(2 * np.pi * df_numeric['week_of_year'] / 52)
    df_numeric = df_numeric.drop(columns=['hour', 'day_of_week', 'month', 'quarter', 'week_of_year'])

    # scaling the numerical features
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df_numeric)
    df_scaled = pd.DataFrame(scaled_array, columns=df_numeric.columns, index=df_numeric.index)

    #reinserting features 
    df_scaled.insert(0, 'datetime_hour', datetime_col)
    df_scaled.insert(1, 'insee_region', insee_col)
    df_scaled.insert(2, 'weather_code', weather_col)
    df_scaled['region_id'] = df_scaled['insee_region']

    #onehot encoding the categorical features that are from the basic dataset
    categorical_cols = ['insee_region', 'weather_code']
    df = pd.get_dummies(df_scaled, columns=categorical_cols, drop_first=False)

    return df, df_numeric, scaler


# Building the sequence for the model

def create_sequences(df, lookback, horizon, group_col, target_col):
    #First we need to sort by region and time
    df.sort_values([group_col, 'datetime_hour'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    exclude_cols = ['datetime_hour', 'region_id']
    numeric_cols = [c for c in df.columns if c not in exclude_cols]

    #setting up the lists to store the targets, input windows nd regions
    X_list, y_list, datetime_list, region_list = [], [], [], []
    #getting the collumn index of the target features from the numeric column list
    target_idx = numeric_cols.index(target_col)

    #looping over each region and convering the numeric columns into a numpy array
    for reg, g in df.groupby(group_col):
        arr = g[numeric_cols].values.astype(np.float32)  #creating a shape for the model  (input window, number of features)
        T = arr.shape[0]  #This will the total length of the input window for a region
        n_sequences = T - lookback - horizon + 1 #here we create the total number of sequences adjusted on the input windows and horizon
        if n_sequences <= 0:
            continue
        #lhere we loop over all the starting points for sequences in the region
        for i in range(n_sequences):
            X_list.append(arr[i:i+lookback])  # with this we create  the past input windows as input
            y_list.append(arr[i+lookback:i+lookback+horizon, target_idx]) # with this we take the horizon as the output
            datetime_list.append(g['datetime_hour'].values[i+lookback:i+lookback+horizon]) # with this we make sure the timestamps are staying at the right targer steps
            region_list.append(reg)

    #Adding it to our list for the model
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    datetime_index = np.array(datetime_list)
    regions_seq = np.array(region_list)
    datetime_target = pd.to_datetime([dt[-1] for dt in datetime_index])

    return X, y, datetime_target, regions_seq, datetime_index


# Splitting the data into training, validation and test

def split_data(X, y, datetime_target, regions_seq):
    #setting the right dates for each split, creating masks for datasplit based on timestamp
    train_mask = (datetime_target >= '2015-01-01') & (datetime_target <= '2017-12-31')
    val_mask = (datetime_target >= '2018-01-01') & (datetime_target <= '2018-12-31')
    test_mask = (datetime_target >= '2019-01-01') & (datetime_target <= '2019-12-31')

    #applying the masks so each sequences has the right timestamps 
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    #creating regions_tests and datetime tests for prediction evaluation metrics in jupyterlab
    regions_test = regions_seq[test_mask]
    datetime_test = datetime_target[test_mask]

    return X_train, y_train, X_val, y_val, X_test, y_test, regions_test, datetime_test, test_mask

#Defining the optuna hyperparameter tuning
def LSTM_tuning(trial, X_train, y_train, X_val, y_val, n_timesteps, n_features, horizon):
    #Setting the ranges for the parameters, based on reasonable ranges to keep it computational managable. 
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_units = trial.suggest_int('n_units', 32, 128)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, 0.5)
    
    activation = 'tanh' # standard activation function for LSTM

    #intializing the models
    model = Sequential()
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        #adding the LSTM layer for the model.
        model.add(LSTM(
            n_units,
            activation=activation,
            return_sequences=return_seq,
            recurrent_dropout=recurrent_dropout,
            input_shape=(n_timesteps, n_features) if i == 0 else None
        ))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate)) #dropout is added here to prevent it from overfitting
    model.add(Dense(horizon)) # this is the output layer, it rpedicts multiple steps ahead in one prediction. 

    # Compiling the model with gradient clipping to stabilize the training
    model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
                  loss='mse', metrics=['mae'])

    #creating early stopping to prevent if from overfitting
    es = EarlyStopping(monitor='val_mae', patience=2, restore_best_weights=True)

    #training the model 
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=1,  # keep 2 for faster tuning
              batch_size=batch_size,
              verbose=0,
              callbacks=[es])

    y_val_pred = model.predict(X_val, verbose=0)
    
    # for hyperparameter tuning to prevent error in the GPU, skipping trials if they produce NaNs
    if np.isnan(y_val_pred).any():
        print("NaNs detected, this trial is skipped.")
        raise optuna.TrialPruned()

    mae_val = mean_absolute_error(y_val, y_val_pred)
    return mae_val


#selecting a subset of the testdata for each month for the feature permutation importance to manage computational size
def make_test_subset_per_month(X_test, y_test, datetime_test, regions_test, week_hours=24*7):
    keep_idx = [] # storing the indices to keep

    #looping over all regions
    for reg in np.unique(regions_test):
        reg_mask = regions_test == reg  # creating the mask to see if the test row belongs to the region
        reg_idx = np.where(reg_mask)[0] # getting the row indices of the full dataset of region
        reg_datetimes = datetime_test[reg_mask] # extracting the timestamps that belongs to the regions rows
        
        months = pd.PeriodIndex(reg_datetimes, freq='M').unique()

        #looping over all months
        for month in months:
            month_mask = pd.PeriodIndex(reg_datetimes, freq='M') == month
            month_idx = reg_idx[month_mask]
            if len(month_idx) >= week_hours:
                keep_idx.extend(month_idx[:week_hours])
            else:
                keep_idx.extend(month_idx)

    #creating the subset arrays for PFI
    X_test_sub = X_test[keep_idx]
    y_test_sub = y_test[keep_idx]
    datetime_test_sub = datetime_test[keep_idx]
    regions_test_sub = regions_test[keep_idx]

    return X_test_sub, y_test_sub, datetime_test_sub, regions_test_sub

#Creating study with storage so GPU does not run out of time limit
def build_and_tune_LSTM(X_train, y_train, X_val, y_val, n_timesteps, n_features, horizon, dataset_name):
    horizon_str = f"{horizon}H"
    study_name = f"LSTM_forecast_{horizon_str}_{dataset_name}"
    storage = f"sqlite:///optuna_LSTM_forecast_{horizon_str}_{dataset_name}.db"

    #Creating the optuna study for better performance
    study = optuna.create_study(direction = 'minimize', study_name=study_name, storage=storage, load_if_exists = True)
    study.optimize(lambda trial: LSTM_tuning(trial, X_train, y_train, X_val, y_val, n_timesteps, n_features, horizon),
                   n_trials=25, show_progress_bar=False)

    best_params = study.best_params
    return best_params


# training the final model with the best hyperparameters optained from optuna
def build_final_model(best_params, X_train, y_train, X_val, y_val, n_timesteps, n_features, horizon, dataset_name):
    horizon_str = f"{horizon}H"
    #initializing the model
    final_model = Sequential()
    #Adding the lstm layers and using the best parameters from the optuna results
    for i in range(best_params['n_layers']):
        return_seq = (i < best_params['n_layers'] - 1)
        final_model.add(LSTM(
            best_params['n_units'],
            activation= 'tanh',
            return_sequences=return_seq,
            recurrent_dropout=best_params['recurrent_dropout'],
            input_shape=(n_timesteps, n_features) if i == 0 else None
        ))
        if best_params['dropout_rate'] > 0:
            final_model.add(Dropout(best_params['dropout_rate'])) #dropout is added here to prevent it from overfitting
    final_model.add(Dense(horizon)) # # this is the output layer, it rpedicts multiple steps ahead in one prediction. 

    #compiling the model using the standard Adam optimizes and MAE and using early stopping of 5 to prevent overfitting
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse', metrics=['mae'])
    es_final = EarlyStopping(monitor='val_mae', patience=5, restore_best_weights=True)

   #creating tracker to log energy usage
    carbon_log_dir = f'./carbon_logs_{horizon_str}_{dataset_name}'
    tracker = CarbonTracker(epochs=50, components='gpu', log_dir=carbon_log_dir)

    tracker.epoch_start()
    #training the model 
    final_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50, # setting how many times training on the dataset
        batch_size=best_params['batch_size'],
        verbose=1,
        callbacks=[es_final] # setting the early stopping
    )
    tracker.epoch_end()
    tracker.stop()
    return final_model

# Creating fucntion to reinverse the scaling of the target feature so it is in the original values

def inverse_target_multi(scaled_array, scaler, target_idx, n_features):
    inv = [] # storing the unscales target values for each horizon
    # creating a temporal array with same numer of features, putting the target features in the target column
    for h in range(scaled_array.shape[1]):
        dummy = np.zeros((scaled_array.shape[0], n_features))
        dummy[:, target_idx] = scaled_array[:, h]
        inv.append(scaler.inverse_transform(dummy)[:, target_idx]) # scaler to convert back to original values
    return np.stack(inv, axis=1)


#Calculating and plotting the permutation feature importance

def plot_permutation_importance_top(model, X_test, y_test, feature_names, horizon=1,
                                    n_repeats=3, one_hot_groups=None, top_n=15,
                                    dataset_name='Test_Data', random_state=42):
    rng = np.random.default_rng(random_state)
    #predicting on the test set ato get the baseline MAE
    baseline_pred = model.predict(X_test, verbose=0)
    baseline_score = mean_absolute_error(y_test[:, :horizon], baseline_pred[:, :horizon])

    #creating the number of features as the input
    n_features = X_test.shape[2]
    importances = np.zeros(n_features)

    #looping over all features for importance and making it computational more managable for GPU
    for f in range(n_features):
        scores = []
        original_feature = X_test[:, :, f].copy()  # saving the original features
        # shuffling the features to perform the permutation
        for _ in range(n_repeats):
            X_test[:, :, f] = rng.permutation(original_feature)  
            y_pred_perm = model.predict(X_test, verbose=0)
            #measure the permutation after shuffling to see increase in error
            scores.append(mean_absolute_error(y_test[:, :horizon], y_pred_perm[:, :horizon]))
        X_test[:, :, f] = original_feature  # restoring back to the original feature values
        importances[f] = np.mean(scores) - baseline_score 
        

    #converting the raw importance values to percentages for better comparison
    importances_pct = 100 * importances / np.sum(importances)
    feature_names_clean = feature_names.copy()
    importances_clean = importances_pct.copy()

    #Aggregating the importance for the one-hot encoded features to get a better overview
    if one_hot_groups is not None:
        for group_name, cols in one_hot_groups.items():
            mask = [i for i, f in enumerate(feature_names_clean) if f in cols]
            if mask:
                avg_importance = importances_clean[mask].mean()
                importances_clean = np.delete(importances_clean, mask)
                feature_names_clean = [f for i, f in enumerate(feature_names_clean) if i not in mask]
                importances_clean = np.append(importances_clean, avg_importance)
                feature_names_clean.append(group_name)

    #selecting the top features
    top_idx = np.argsort(importances_clean)[-top_n:]
    top_idx = top_idx[np.argsort(importances_clean[top_idx])[::-1]]

    # creating a table for better interpretation and comparison
    importance_table = pd.DataFrame({
        "Feature": [feature_names_clean[i] for i in top_idx],
        "Importance (%)": np.round(importances_clean[top_idx], 3)
    })

    print("\nTop Feature Importances:")
    print(importance_table.to_string(index=False))

    # Creating the permutation feature plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_table["Feature"], importance_table["Importance (%)"])
    plt.xlabel("Importance (%)", fontsize=11)
    plt.title(f"Top {top_n} Feature Importances: Horizon {horizon}h", fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # saving the plot for usage after the GPU finished the job
    plot_file = f'LSTM_{horizon}H_{dataset_name}_feature_importance.png'
    csv_file = f'LSTM_{horizon}H_{dataset_name}_feature_importance_table.csv'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    importance_table.to_csv(csv_file, index=False)
    return importance_table

#Creating the shap feature importance
def plot_shap_importance_percentage(
    model, X_test, feature_names,
    n_samples=200,
    plot_title="SHAP Feature Importance in (%)",
    save_path='SHAP_importance_percentage.png',
    top_n_bar=15,
    one_hot_groups=None  
):

    #selecting a random subset of samples for shap to keep it computational managable
    n_background = min(50, len(X_test))
    background_idx = np.random.choice(len(X_test), size=n_background, replace=False)
    explain_idx = np.random.choice(len(X_test), size=n_samples, replace=False)

    X_background = X_test[background_idx]
    X_sub = X_test[explain_idx]

    # create SHAP explainer using Gradientexplainer for the model
    explainer = shap.GradientExplainer(model, X_background)
    # compute SHAP values for the selected samples
    shap_output = explainer.shap_values(X_sub)
    shap_values = shap_output[0] if isinstance(shap_output, list) else shap_output
    shap_values = np.array(shap_values)

    #fixing and preventing error of output shapes to make sure GPU keeps running
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.ndim == 2:
        shap_values = shap_values[:, np.newaxis, :]

    #getting the SHAP importance of each feature
    shap_importance = np.mean(np.abs(shap_values), axis=(0, 1))

    shap_importance_clean = shap_importance.copy()
    feature_names_clean = feature_names.copy()

    #combining the one-hot encoded columns back into a feature
    if one_hot_groups is not None:
        for group_name, cols in one_hot_groups.items():
            #finding which feature belongs where
            mask = [i for i, f in enumerate(feature_names_clean) if f in cols]
            #averaging the importance of the one-hot encoded features
            if mask:
                avg_importance = shap_importance_clean[mask].mean()
                #removing the indivudal one-hot features from the list
                shap_importance_clean = np.delete(shap_importance_clean, mask)
                feature_names_clean = [f for i, f in enumerate(feature_names_clean) if i not in mask]
                #adding the 
                shap_importance_clean = np.append(shap_importance_clean, avg_importance)
                feature_names_clean.append(group_name)


    #converting the features into percentages for better comparison 
    shap_importance_pct = 100 * shap_importance_clean / np.sum(shap_importance_clean)

    top_idx_bar = np.argsort(shap_importance_pct)[-top_n_bar:]
    feature_names_sorted = np.array(feature_names_clean)[top_idx_bar]
    shap_importance_sorted = shap_importance_pct[top_idx_bar]

    #Creating the plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names_sorted[::-1], shap_importance_sorted[::-1])
    plt.xlabel("SHAP contribution (%)")
    plt.title(plot_title + f": Top SHAP {top_n_bar} Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# creating the main to run the script in the GPU
def main():
    #Setting the parameters so i can change parameters for each experiment
    dataset_name = "Additional_Data"
    df, df_numeric, scaler = load_and_preprocess_data()
    target_idx = df_numeric.columns.get_loc('conso_elec_mw')
    lookback = 24           # lookback of 24 means, looking back 1 day per row
    horizon = 1             # horizon of 1, predicting 1 hour ahead
    group_col = 'region_id'
    target_col = 'conso_elec_mw'

    X, y, datetime_target, regions_seq, datetime_index = create_sequences(df, lookback, horizon, group_col, target_col)
    X_train, y_train, X_val, y_val, X_test, y_test, regions_test, datetime_test, test_mask = split_data(X, y, datetime_target, regions_seq)

    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    best_params = build_and_tune_LSTM(X_train, y_train, X_val, y_val, n_timesteps, n_features, horizon, dataset_name)
    final_model = build_final_model(best_params, X_train, y_train, X_val, y_val, n_timesteps, n_features, horizon, dataset_name)

    y_pred_test = final_model.predict(X_test)
    y_pred_test_inv = inverse_target_multi(y_pred_test, scaler, target_idx, df_numeric.shape[1])
    y_test_inv = inverse_target_multi(y_test, scaler, target_idx, df_numeric.shape[1])

    feature_names = [c for c in df.columns if c not in ['datetime_hour', 'region_id']]

    one_hot_groups = {
        'region': [c for c in feature_names if 'insee_region_' in c],
        'weather': [c for c in feature_names if 'weather_code_' in c],
        'school_zone': [c for c in feature_names if 'school_zone_' in c],
        'vacation_name': [c for c in feature_names if 'vacation_name_' in c],
        'holiday_name': [c for c in feature_names if 'holiday_name_' in c],
        'season': [c for c in feature_names if 'season_' in c],
        'hour': [c for c in feature_names if 'sin_hour' in c or 'cos_hour' in c],
        'day_of_week': [c for c in feature_names if 'sin_day' in c or 'cos_day' in c],
        'month': [c for c in feature_names if 'sin_month' in c or 'cos_month' in c],
        'quarter': [c for c in feature_names if 'sin_quarter' in c or 'cos_quarter' in c],
        'week_of_year': [c for c in feature_names if 'sin_week' in c or 'cos_week' in c]
    }

    plot_shap_importance_percentage(
        model=final_model,
        X_test=X_test,
        feature_names=feature_names,     
        top_n_bar=15,
        one_hot_groups=one_hot_groups, 
        save_path='SHAP_LSTM_Additional_Data_importance_1h.png'
    )



    X_test_sub, y_test_sub, datetime_test_sub, regions_test_sub = make_test_subset_per_month(
        X_test, y_test, datetime_test, regions_test, week_hours=24*7)
    plot_permutation_importance_top(final_model, X_test_sub, y_test_sub, feature_names, horizon=horizon,
                                    one_hot_groups=one_hot_groups, top_n=15, dataset_name=dataset_name, n_repeats = 1, random_state = SEED)
    all_datetimes = np.concatenate(datetime_index[test_mask])

    #creating CSV to evaluate and make plots in jupyterlab with the model predictions after running in GPU
    eval_df = pd.DataFrame({
        'Model_name': f'LSTM_AD_{horizon}H',
        'datetime': pd.to_datetime(all_datetimes),
        'region': np.repeat(regions_test, horizon),
        'horizon': np.tile(np.arange(1, horizon + 1), len(datetime_test)),
        'y_true': y_test_inv.flatten(),
        'y_pred': y_pred_test_inv.flatten()
    })

    csv_file = f'GPU_Predictions_LSTM_{horizon}H_{dataset_name}.csv'
    eval_df.to_csv(csv_file, index=False)
    print('Finished testing')

if __name__ == "__main__":
    main()

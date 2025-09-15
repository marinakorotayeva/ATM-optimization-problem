import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
import signal
from contextlib import contextmanager

warnings.filterwarnings('ignore')

# Set matplotlib parameters to handle date formatting better
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['figure.max_open_warning'] = 50

# Create directories if they don't exist
os.makedirs("Prediction models/LSTM/results", exist_ok=True)
os.makedirs("Prediction models/LSTM/predictions", exist_ok=True)
os.makedirs("Prediction models/LSTM/graphs/actual_vs_predicted", exist_ok=True)
os.makedirs("Prediction models/LSTM/graphs/residuals", exist_ok=True)
#os.makedirs("Prediction models/LSTM/models", exist_ok=True)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Timeout handling
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Load prepared data for cross-validation
try:
    with open("Prediction models/Data/prepared_data.pkl", "rb") as f:
        prepared_data = pickle.load(f)
except FileNotFoundError:
    print("Error: Prepared data file not found. Please run data_preparation.py first.")
    print("Expected file: Prediction models/Data/prepared_data.pkl")
    exit(1)

# Load cross-validator
try:
    with open("Prediction models/Data/tscv.pkl", "rb") as f:
        tscv = pickle.load(f)
except FileNotFoundError:
    print("Error: Cross-validator file not found. Please run data_preparation.py first.")
    print("Expected file: Prediction models/Data/tscv.pkl")
    exit(1)

# Function to calculate SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero when both true and pred are zero
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    return 100 * np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])

# Function to create optimized LSTM model
def create_optimized_lstm_model(input_shape, units=64, dropout_rate=0.3, learning_rate=0.0005):
    model = Sequential([
        Bidirectional(LSTM(units, activation='tanh', return_sequences=True, 
                          kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
                     input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units//2, activation='tanh', return_sequences=False),  # Changed to False for simplicity
        Dropout(dropout_rate),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(16, activation='relu'),  # Added another layer
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),  # Fixed: Use Huber class instead of string
        metrics=['mae', 'mse']
    )
    
    return model

# Function to prepare sequences for LSTM with multiple time steps
def create_sequences(X, y, time_steps=14):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# Function for simple hyperparameter optimization
def optimize_hyperparameters(X_train, y_train, input_shape):
    """Simple hyperparameter optimization without KerasTuner"""
    best_score = float('inf')
    best_params = {'units': 32, 'dropout': 0.2, 'lr': 0.001}  # Start with simpler defaults
    
    # Test a few configurations (simplified without full grid search)
    configurations = [
        {'units': 32, 'dropout': 0.2, 'lr': 0.001},
        {'units': 64, 'dropout': 0.3, 'lr': 0.0005},
        {'units': 32, 'dropout': 0.3, 'lr': 0.0005},
        {'units': 64, 'dropout': 0.2, 'lr': 0.001}
    ]
    
    for config in configurations:
        try:
            model = create_optimized_lstm_model(
                input_shape, config['units'], config['dropout'], config['lr']
            )
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
            
            # Use timeout for hyperparameter optimization
            try:
                with time_limit(120):  # 2 minutes per config
                    history = model.fit(
                        X_train, y_train,
                        epochs=20,  # Shorter for faster testing
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stop, reduce_lr],
                        verbose=0
                    )
                    
                    val_loss = min(history.history['val_loss'])
                    if val_loss < best_score:
                        best_score = val_loss
                        best_params = config.copy()
                        
            except TimeoutException:
                print(f"  Configuration timed out: {config}")
                continue
                
        except Exception as e:
            print(f"  Configuration failed: {config}, error: {e}")
            continue
    
    return best_params

# Store results and predictions
combined_results = []
prediction_rows = []

# Process each ATM
for atm_num, data_dict in prepared_data.items():
    print(f"\n--- Processing ATM {atm_num} ---")
    
    X = data_dict['X']
    y = data_dict['y']
    dates = data_dict['dates']
    features = data_dict['features']
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {features}")
    
    # Use RobustScaler for better handling of outliers
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Create sequences for LSTM with optimized time steps
    time_steps = 14
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
    dates_seq = dates[time_steps:]  # Adjust dates to match sequences
    
    print(f"Sequences created: {X_seq.shape[0]} samples with {time_steps} time steps")
    
    # Manual cross-validation
    print("Performing cross-validation with time series CV...")
    cv_rmse_scores = []
    cv_mae_scores = []
    cv_smape_scores = []
    cv_models = []
    
    fold = 1
    for train_index, test_index in tscv.split(X_seq):
        # Ensure we have enough data for this fold
        if len(train_index) < 20 or len(test_index) < 10:
            print(f"Skipping fold {fold} - insufficient data")
            fold += 1
            continue
            
        X_train, X_test = X_seq[train_index], X_seq[test_index]
        y_train, y_test = y_seq[train_index], y_seq[test_index]
        
        # Use default parameters for first fold, optimize for subsequent ones
        if fold == 1:
            try:
                with time_limit(300):  # 5 minutes for hyperparameter optimization
                    best_params = optimize_hyperparameters(X_train, y_train, (time_steps, X.shape[1]))
                print(f"  Best hyperparameters: {best_params}")
            except TimeoutException:
                print("  Hyperparameter optimization timed out, using defaults")
                best_params = {'units': 32, 'dropout': 0.2, 'lr': 0.001}
        else:
            # Reuse the same parameters for consistency
            pass
        
        # Create and train optimized LSTM model
        model = create_optimized_lstm_model(
            (time_steps, X.shape[1]), 
            best_params['units'], 
            best_params['dropout'], 
            best_params['lr']
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=8, 
            restore_best_weights=True, 
            verbose=0
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6, 
            verbose=0
        )
        
        print(f"  Training fold {fold}...")
        
        try:
            with time_limit(300):  # 5 minutes per fold
                history = model.fit(
                    X_train, y_train,
                    epochs=80,  # Reduced from 100
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                
                # Predict and inverse transform
                y_pred_fold_scaled = model.predict(X_test, verbose=0).flatten()
                y_pred_fold = scaler_y.inverse_transform(y_pred_fold_scaled.reshape(-1, 1)).flatten()
                y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                cv_rmse_scores.append(np.sqrt(mean_squared_error(y_test_actual, y_pred_fold)))
                cv_mae_scores.append(mean_absolute_error(y_test_actual, y_pred_fold))
                cv_smape_scores.append(symmetric_mean_absolute_percentage_error(y_test_actual, y_pred_fold))
                cv_models.append(model)
                
                print(f"  Fold {fold} completed - RMSE: {cv_rmse_scores[-1]:.2f}, SMAPE: {cv_smape_scores[-1]:.2f}%")
                
        except TimeoutException:
            print(f"  Fold {fold} timed out - skipping")
        except Exception as e:
            print(f"  Fold {fold} failed with error: {e}")
        
        fold += 1
    
    if not cv_models:
        print(f"Warning: No valid models trained for ATM {atm_num}")
        continue
        
    # Select best model from CV
    best_fold_idx = np.argmin(cv_rmse_scores)
    best_model = cv_models[best_fold_idx]
    
    # Save the best model
    #best_model.save(f"Prediction models/LSTM/models/ATM_{atm_num}_best_model.h5")
    
    # Train final model on full dataset with best hyperparameters
    print("Training final model on full dataset...")
    final_model = create_optimized_lstm_model(
        (time_steps, X.shape[1]), 
        best_params['units'], 
        best_params['dropout'], 
        best_params['lr']
    )
    
    try:
        with time_limit(600):  # 10 minutes for final training
            history = final_model.fit(
                X_seq, y_seq,
                epochs=100,  # Reduced from 150
                batch_size=32,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=0)
                ],
                verbose=0
            )
            
            # Save final model
            #final_model.save(f"Prediction models/LSTM/models/ATM_{atm_num}_final_model.h5")
            
            # Predict on full dataset
            y_pred_scaled = final_model.predict(X_seq, verbose=0).flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_actual = scaler_y.inverse_transform(y_seq.reshape(-1, 1)).flatten()
            
            # Calculate metrics on the entire dataset
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            smape = symmetric_mean_absolute_percentage_error(y_actual, y_pred)
            
            # Calculate residuals
            residuals = y_actual - y_pred
            
            # Store results
            combined_results.append({
                'ATM_NUMBER': atm_num,
                'CV_RMSE_Mean': np.mean(cv_rmse_scores) if cv_rmse_scores else np.nan,
                'CV_RMSE_Std': np.std(cv_rmse_scores) if cv_rmse_scores else np.nan,
                'CV_MAE_Mean': np.mean(cv_mae_scores) if cv_mae_scores else np.nan,
                'CV_MAE_Std': np.std(cv_mae_scores) if cv_mae_scores else np.nan,
                'CV_SMAPE_Mean': np.mean(cv_smape_scores) if cv_smape_scores else np.nan,
                'CV_SMAPE_Std': np.std(cv_smape_scores) if cv_smape_scores else np.nan,
                'MAE_Full': mae,
                'RMSE_Full': rmse,
                'SMAPE_Full': smape,
                'Best_Units': best_params['units'],
                'Best_Dropout': best_params['dropout'],
                'Best_Learning_Rate': best_params['lr'],
                'Time_Steps': time_steps,
                'Total_Samples': len(X_seq),
                'Features_Used': len(features)
            })
            
            print(f"CV Results - Mean RMSE: {np.mean(cv_rmse_scores):.2f} ± {np.std(cv_rmse_scores):.2f}")
            print(f"CV Results - Mean SMAPE: {np.mean(cv_smape_scores):.2f}% ± {np.std(cv_smape_scores):.2f}%")
            print(f"Full Dataset - RMSE: {rmse:.2f}, SMAPE: {smape:.2f}%")
            print(f"Best Hyperparameters - Units: {best_params['units']}, Dropout: {best_params['dropout']}, LR: {best_params['lr']}")
            
            # Store predictions with dates
            predictions_df = pd.DataFrame({
                'ATM_NUMBER': atm_num,
                'DATE': dates_seq,
                'Actual': y_actual,
                'Predicted': y_pred,
                'Residual': residuals,
                'Absolute_Error': np.abs(residuals),
                'Percentage_Error': np.abs(residuals) / (y_actual + 1e-10) * 100,  # Avoid division by zero
                'Fold_Info': '10-Fold_CV'
            })
            
            prediction_rows.append(predictions_df)
            
            # Create enhanced visualizations
            # 1. Time Series Residuals Plot
            plt.figure(figsize=(16, 10))
            
            # Residuals plot
            plt.subplot(2, 1, 1)
            plt.plot(dates_seq, residuals, color='green', linestyle='-', alpha=0.7, label='Residual Trend')
            plt.scatter(dates_seq, residuals, color='green', alpha=0.5, s=15, label='Residuals')
            plt.axhline(y=0, color='r', linestyle='--', label='Zero Error', linewidth=2)
            plt.fill_between(dates_seq, -np.std(residuals), np.std(residuals), alpha=0.2, color='gray', label='±1 Std Dev')
            
            plt.title(f'ATM {atm_num} - LSTM Residuals Analysis\nRMSE: {rmse:.0f}, SMAPE: {smape:.1f}%')
            plt.xlabel('Date')
            plt.ylabel('Residuals (Actual - Predicted)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Improved date formatting
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.gcf().autofmt_xdate()
            
            # Residual distribution
            plt.subplot(2, 1, 2)
            plt.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
            plt.title('Residual Distribution')
            plt.xlabel
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"Prediction models/LSTM/graphs/residuals/ATM_{atm_num}_residuals_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Enhanced actual vs predicted plot
            plt.figure(figsize=(16, 12))
            
            # Main time series plot
            plt.subplot(2, 1, 1)
            plt.plot(dates_seq, y_actual, label='Actual', color='blue', alpha=0.8, linewidth=2)
            plt.plot(dates_seq, y_pred, label='Predicted', color='red', linestyle='-', alpha=0.7, linewidth=1.5)
            plt.fill_between(dates_seq, y_pred - rmse, y_pred + rmse, alpha=0.2, color='red', label='Prediction ± RMSE')
            
            plt.title(f'ATM {atm_num} - LSTM Actual vs Predicted\nRMSE: {rmse:.0f}, SMAPE: {smape:.1f}%')
            plt.xlabel('Date')
            plt.ylabel('Withdrawals')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Improved date formatting
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.gcf().autofmt_xdate()
            
            # Scatter plot of actual vs predicted
            plt.subplot(2, 1, 2)
            plt.scatter(y_actual, y_pred, alpha=0.6, color='purple')
            max_val = max(np.max(y_actual), np.max(y_pred))
            plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
            plt.xlabel('Actual Withdrawals')
            plt.ylabel('Predicted Withdrawals')
            plt.title('Actual vs Predicted Scatter Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"Prediction models/LSTM/graphs/actual_vs_predicted/ATM_{atm_num}_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Training history plot
            plt.figure(figsize=(12, 8))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'ATM {atm_num} - Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"Prediction models/LSTM/graphs/ATM_{atm_num}_training_history.png", dpi=300, bbox_inches='tight')
            plt.close()
            
    except TimeoutException:
        print("Final model training timed out")
    except Exception as e:
        print(f"Final model training failed with error: {e}")
    
    # Save checkpoint after each ATM
    checkpoint_data = {
        'completed_atms': [atm_num],
        'combined_results': combined_results,
        'prediction_rows': prediction_rows
    }
    
    #with open(f"Prediction models/LSTM/checkpoint_ATM_{atm_num}.pkl", 'wb') as f:
        #pickle.dump(checkpoint_data, f)
    
    #print(f"Completed ATM {atm_num} and saved checkpoint")

# Save final results
if combined_results:
    results_df = pd.DataFrame(combined_results)
    results_df.to_csv("Prediction models/LSTM/results/lstm_cv_results.csv", index=False)
    print("\nCross-validation results saved to: Prediction models/LSTM/results/lstm_cv_results.csv")

if prediction_rows:
    all_predictions = pd.concat(prediction_rows, ignore_index=True)
    all_predictions.to_csv("Prediction models/LSTM/predictions/lstm_all_predictions.csv", index=False)
    print("All predictions saved to: Prediction models/LSTM/predictions/lstm_all_predictions.csv")

    # Save summary statistics
   #summary_stats = all_predictions.groupby('ATM_NUMBER').agg({
        #'Absolute_Error': ['mean', 'std', 'min', 'max'],
        #'Percentage_Error': ['mean', 'std', 'min', 'max'],
        #'Residual': ['mean', 'std']
    #}).round(2)
    
    #summary_stats.to_csv("Prediction models/LSTM/results/lstm_summary_statistics.csv")
    #print("Summary statistics saved to: Prediction models/LSTM/results/lstm_summary_statistics.csv")

print("\nLSTM modeling completed for all ATMs!")

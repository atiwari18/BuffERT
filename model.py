import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line


# List of stocks to analyze
#Retrieve stock data and feature engineering


def get_stock_data(stock_list, start_date, end_date):
   
    all_stocks = []

    print("here")
    for stock in stock_list:
        data = yf.Ticker(stock).history(start=start_date, end=end_date)
        data['Ticker'] = stock
        #print(data.head())
        data_dropped = data.drop(columns = ['Dividends', 'Stock Splits'])
        all_stocks.append(data_dropped)


    

    df = pd.concat(all_stocks)
    df_sorted = df.sort_values(by=['Ticker', 'Date'])
    df_sorted = df_sorted.reset_index()

    
    df_sorted['Return'] = df_sorted.groupby('Ticker')['Close'].pct_change()

    df_sorted['Price Change'] = (df_sorted['Return'] > 0).astype(int)


    df_sorted = df_sorted.dropna().reset_index(drop=True)
    df_sorted = df_sorted.drop(columns = ['Open', 'High', 'Low', 'Volume', 'Return'])
  
   
    return df_sorted


def preprocess_stock(data):

    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(data[[ 'Return', 'MA7', 'MA14', 'MA30', 'RSI', 'MACD', 'Signal_Line', 'Rolling_Volatility']])
    scaled_df = pd.DataFrame(scaled_data, columns=[ 'Return', 'MA7', 'MA14', 'MA30', 'RSI', 'MACD', 'Signal_Line', 'Rolling_Volatility'])
    scaled_df['Price Change'] = data['Price Change'].values

    return scaled_df


#create sequences and targets for each stock
#returns a list of sequences, list of targets respectively, for a stock
def create_sequences(data, sequence_length):
    
    seqs, targets = [], []
    cols = [ 'Return', 'MA7', 'MA14', 'MA30', 'RSI', 'MACD', 'Signal_Line', 'Rolling_Volatility']
    print("Length of data: ", len(data))
    for i in range((len(data) - sequence_length )):
        
        sequence = data[cols].iloc[i:i+sequence_length].values

        
        target =  data['Price Change'].iloc[i+sequence_length]
        seqs.append(sequence)
        targets.append(target)

    return seqs, targets




def create_train_validation_sets(data, stock_list, sequence_length):

    #takes each individual stock data, sorts by date asc, drops non-numerical cols,

        X_train, X_test, Y_train, Y_test = [], [], [], [] #holds seq/targets for every stock
        i = 0
        for stock in stock_list:

            stock_data = data[data['Ticker'] == stock]
            sorted_by_date = stock_data.sort_values('Date')
            

            sorted_by_date['Return'] = sorted_by_date['Close'].pct_change()
            sorted_by_date['MA7'] = sorted_by_date['Close'].rolling(window=7).mean()
            sorted_by_date['MA14'] = sorted_by_date['Close'].rolling(window=14).mean()
            sorted_by_date['MA30'] = sorted_by_date['Close'].rolling(window=30).mean()
            sorted_by_date['Rolling_Volatility'] = sorted_by_date['Return'].rolling(window=30).std()
            sorted_by_date['RSI'] = compute_rsi(sorted_by_date['Close'])
            sorted_by_date['MACD'], sorted_by_date['Signal_Line'] = compute_macd(sorted_by_date['Close'])
            
            
            sorted_by_date = sorted_by_date.dropna().reset_index(drop=True)
            
            clean_data = sorted_by_date.drop(columns = ['Date', 'Ticker', 'Close'])
            
            
            scaled_data = preprocess_stock(clean_data)
            #print(scaled_data.head())
            #print(scaled_data.tail())
            #corr = scaled_data.corr()
            
          
            seqs, targets = create_sequences(scaled_data, sequence_length)
            #print(stock)
            print(len(seqs))
        ##print(seqs[0].shape)
        #print(len(targets))
        #print(targets)
        
            split = int (0.8 * len(seqs))


            X_train.extend(seqs[:split])
            X_test.extend(seqs[split:])

            Y_train.extend(targets[:split])
            Y_test.extend(targets[split:])

        
        

        #rehape the inputs for   LSTM model
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)
        print(X_train.shape)
        print(len(Y_train))

        return X_train, X_test, Y_train, Y_test





def train_BiLSTM(X_train, X_test, Y_train, Y_test):

        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2]))))
        model.add(Dropout(0.3))

        model.add(Bidirectional(LSTM(64, return_sequences=False)))  # Final LSTM, no sequences returned
        model.add(Dropout(0.3))

        model.add(Dense(32, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        

        weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
        class_weights = {0: weights[0], 1: weights[1]}

        stats = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test), verbose =1, class_weight=class_weights)


       


        loss, accuracy = model.evaluate(X_test, Y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int).flatten()
        y_labels = Y_test.flatten()   
        precision = precision_score(y_labels, y_pred)
        recall = recall_score(y_labels, y_pred)
        f1 = f1_score(y_labels, y_pred)
        accuracy = accuracy_score(y_labels, y_pred)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:   {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        
        

        return model


def forecast(model, data, stock, days, sequence_length = 30):
            
            forecasts = []
            stock_data = data[data['Ticker'] == stock]
            sorted_by_date = stock_data.sort_values('Date')
            
            
            for day in range(days):
                print(day)
                sorted_by_date['Return'] = sorted_by_date['Close'].pct_change()
                sorted_by_date['MA7'] = sorted_by_date['Close'].rolling(window=7).mean()
                sorted_by_date['MA14'] = sorted_by_date['Close'].rolling(window=14).mean()
                sorted_by_date['MA30'] = sorted_by_date['Close'].rolling(window=30).mean()
                sorted_by_date['Rolling_Volatility'] = sorted_by_date['Return'].rolling(window=30).std()
                sorted_by_date['RSI'] = compute_rsi(sorted_by_date['Close'])
                sorted_by_date['MACD'], sorted_by_date['Signal_Line'] = compute_macd(sorted_by_date['Close'])
                

                sorted_by_date = sorted_by_date.dropna().reset_index(drop=True)

                pred_window = sorted_by_date.tail(sequence_length) 

                processed = preprocess_stock(pred_window)
                train = processed.drop(columns=['Price Change']).values
                print(train.shape())
                input = train.reshape(1, sequence_length, train.shape[1])

                probability = model.predict(input)[0][0]
                prediction = int(probability > 0.5)
                

                last_close = sorted_by_date['Close'].iloc[-1]

                forecast_date = pd.date_range(start=sorted_by_date['Date'].iloc[-1] + pd.Timedelta(days=1), periods=1, freq='B')[0]
                forecasts.append(prediction)
                next_close = last_close * (1 + (0.005 if prediction else -0.005))

                sorted_by_date = pd.concat([
                    sorted_by_date,
                    pd.DataFrame([{
                    'Date': forecast_date,
                    'Close': next_close,
                    'Price Change': prediction, 
                    'Ticker': sorted_by_date['Ticker'].iloc[-1]
                }])
            ], ignore_index=True)
    
     
            return forecasts


def main():

    stock_list =  ['ABBV', 'ADBE', 'ADI', 'ABNB', 'ZTS',
                    'XOM', 'JNJ', 'ACN', 'CMCSA', 'LMT',
                    'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    

    train_data = get_stock_data(stock_list, '2023-01-01', '2025-01-04')

    X_train, X_test, Y_train, Y_test = create_train_validation_sets(train_data, stock_list, sequence_length=30)

    model = train_BiLSTM(X_train, X_test, Y_train, Y_test)

 

    


if __name__ == "__main__":
    main()


    



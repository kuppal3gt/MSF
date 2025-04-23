# Load necessary libraries
library(shiny)
library(ggplot2)
#library(prophet)
library(forecast)
#library(keras)
library(reticulate)
library(dplyr)
library(plyr)
library(quantmod)
library(lubridate)
library(tidyr)
library(shinyFeedback)
library(plotly)
library(reticulate)
#install_miniconda()
#use_python("/opt/anaconda3/bin/") #"C:\\Users\\karan\\anaconda3\\")
#source_python("C:\\Users/karan/Downloads/call_alpaca.py")
#source_python("C:\\Users/karan/Downloads/call_dynamic_alpaca.py")
#& MA5.Check<0 & MA50.Check<0 && Price.Status=="Oversold" 
#source("/Users/karanuppal/Library/CloudStorage/OneDrive-Personal/mystockforecast_scripts/current/UtilityFunctions_trading.R")
#source_python("/Users/karanuppal/Library/CloudStorage/OneDrive-Personal/mystockforecast_scripts/current/call_alpaca_dev_04172025.py") 
# Use the appropriate conda environment
#library(reticulate)
#conda_install("r-reticulate", c("pandas","ta","alpaca-trade-api","pytz","bs4","time","datetime","zoneinfo","textblob","numpy","nltk","re","scikit-learn","joblib"),channel="conda-forge")

#use_condaenv("r-reticulate", required = TRUE)
# Install pip-only package into conda env
#py_install("alpaca-py", pip = TRUE, envname = "r-reticulate")
#py_install("ta", pip = TRUE, envname = "r-reticulate")
#py_install("nest_asyncio", pip = TRUE, envname = "r-reticulate")
#py_install(c("bs4","textblob","scikit-learn"), pip = TRUE, envname = "r-reticulate")
#use_virtualenv("./myenv",required=TRUE)

# Automatically use shinyapps.io's Python
#use_python(Sys.getenv("RETICULATE_PYTHON", unset = "python3"))
#use_virtualenv("~/myenv", required = TRUE) 
source("UtilityFunctions_trading.R")
source_python("call_alpaca_dev_04172025.py") 

calculate_lstm_forecast<-function(df){
  
  series <- as.numeric(df)
  series_scaled <- scales::rescale(series)
  
  window_size <- 20
  
  if (length(series_scaled) <= window_size) {
    stop("Not enough data for the selected window_size. Try reducing it or expanding your time range.")
  }
  
  X <- array(0, dim = c(length(series_scaled) - window_size, window_size, 1))
  y <- array(0, dim = c(length(series_scaled) - window_size))
  
  for (i in 1:(length(series_scaled) - window_size)) {
    X[i,,1] <- series_scaled[i:(i + window_size - 1)]
    y[i] <- series_scaled[i + window_size]
  }
  
  model_lstm <- keras_model_sequential()
  model_lstm %>% 
    layer_lstm(units = 50, input_shape = c(window_size, 1), return_sequences = FALSE) %>%
    layer_dense(units = 1)

  
  model_lstm %>% compile(
    optimizer = 'adam',
    loss = 'mean_squared_error'
  )
  
  model_lstm %>% fit(X, y, epochs = 20, batch_size = 32)
  
  last_window <- array(series_scaled[(length(series_scaled) - window_size + 1):length(series_scaled)], dim = c(1, window_size, 1))
  lstm_pred_scaled <-predict(model_lstm,last_window)
  lstm_pred <- rescale(lstm_pred_scaled, to = range(series))
  
}

# Create lagged features (window of 5 days)
lag_features <- function(df, k = 5) {
  for (i in 1:k) {
    df[[paste0("lag", i)]] <- lag(df$close, i)
  }
  #df <- na.omit(df)
  return(df)
}

run_predictoin<-function(df){
  
 
  
  data_lagged <- lag_features(df, 5)
  
  predictors <- c("lRSI_14", "lfastK", "lfastK_5", "lfastD", "lvolume_change_ratio", "lvolume_change", "lROC_5", "lROC_10",  "l5_MA", "lmomentum_1", "lmomentum_3", "lmomentum_5",
                  "lMACD_histogram")
  # Split into train/test (e.g., last 100 days as test)
  split_index <- nrow(data_lagged) - 10
  train_data <- data_lagged[1:split_index, predictors]
  test_data <- data_lagged[(split_index + 1):nrow(data_lagged), predictors]
  
  train_data<-na.omit(train_data)
  test_data<-na.omit(test_data)
  # Define predictors and target
  
   #setdiff(names(train_data), c("Date", "Price"))
  target <- "close"
  
  # 1. Random Forest Regression
  set.seed(123)
  rf_model <- randomForest(
    x = train_data[, predictors],
    y = train_data$close,
    ntree = 500
  )
  
  rf_preds <- predict(rf_model, newdata = test_data[, predictors])
  
  # 2. Support Vector Regression (SVR)
  svr_model <- svm(
    Price ~ ., data = train_data[, c(predictors, target)],
    kernel = "radial"
  )
  
  svr_preds <- predict(svr_model, newdata = test_data[, predictors])
  
  # Compare predictions with actual values
  results <- data.frame(
    Date = test_data$Date,
    Actual = test_data$Price,
    RF = rf_preds,
    SVR = svr_preds
  )
  
  results_long <- results %>%
    pivot_longer(cols = c("RF", "SVR", "Actual"), names_to = "Model", values_to = "Price")
  
  return(results_long)
  
}

is_valid_symbol <- function(symbol) {
  tryCatch({
    suppressWarnings(
      getSymbols(Symbols = symbol, src = "yahoo", auto.assign = FALSE)
    )
    TRUE
  }, error = function(e) {
    FALSE
  })
}
#calculate_lstm_forecast(df$close)
# Define UI for application
ui <- fluidPage(
  titlePanel("Stock Price Forecasts and Ratings"),
  useShinyFeedback(),
  sidebarLayout(
    sidebarPanel(
      textInput("symbol",
                  "Choose a symbol:"
                 ),
      #dateInput("selected_date", "Select a date:", value = Sys.Date()),
      #verbatimTextOutput("date_output"),
      actionButton("update", "Submit"),width = 2
    ),
    
    mainPanel(
      plotlyOutput("pricePlot"),
      tableOutput("ratingTable")
    )
  )
)

# Define server logic
server <- function(input, output) {
  
  symbol_check<-eventReactive(input$update, {
    # Get the data (you'll need to replace this with your own data fetching and processing logic)
    #data <- get_data(input$industry)
   # df<-fetch_and_compute_indicators(input$symbol,Sys.Date()-360,Sys.Date(),scorethresh = 7)%>%mutate(Symbol=input$symbol)
    # Calculate forecasts
    #prophet_forecast <- calculate_prophet_forecast(df$close)
    #lstm_forecast <- calculate_lstm_forecast(df$close)
    #print(head(df))
    
    req(input$symbol)
    sym <- (trimws(input$symbol))
    if (!is_valid_symbol(input$symbol)) {
      showFeedbackDanger("symbol", paste(sym, "is NOT a valid stock symbol ❌"))
      return(NULL)
    } else {
      hideFeedback("symbol")
      return(sym)
    }
  })
  observeEvent(input$update,{
    req(symbol_check())
    symbol=symbol_check()
    sym <- toupper(trimws(symbol))
    if (!is_valid_symbol(input$symbol)) {
      showFeedbackDanger("symbol", paste(sym, "is NOT a valid stock symbol ❌"))
    }
    
    df1<-fetch_and_compute_indicators(symbol,Sys.Date(),Sys.Date(),timeframe = "Minute",scorethresh = 7)%>%mutate(Symbol=symbol,Rating=ifelse(is.na(signal)==TRUE,"HOLD","BUY"),
                                                                                                   Score=ifelse(buy_score<4,"Low",ifelse(buy_score<8,"Moderate","High")))
    df=df1
    df2<-fetch_and_compute_indicators(symbol,Sys.Date()-120,Sys.Date(),timeframe = "Day",scorethresh = 7)%>%mutate(Symbol=symbol,Rating=ifelse(is.na(signal)==TRUE,"HOLD","BUY"),
                                                                                                                  Score=ifelse(buy_score<4,"Low",ifelse(buy_score<8,"Moderate","High")))
    
    # Create the forecast plot
    output$pricePlot <- renderPlotly({
     # ggplot() +
      ##  geom_line(data = prophet_forecast, aes(x = Date, y = Forecast), color = "blue") +
      #  geom_line(data = lstm_forecast, aes(x = Date, y = Forecast), color = "red") +
      #  geom_line(data = arima_forecast, aes(x = Date, y = Forecast), color = "green") +
      #  labs(title = paste("Price forecast for", input$industry),
       #      x = "Date",
       #      y = "Forecasted Price")
      library(forecast)
      
      fit_arima <- auto.arima(df$close[(1):nrow(df)])
      forecast_arima <- forecast(fit_arima, h = 30)
      #arima_forecast <- calculate_arima_forecast(df$close)
      
      library(prophet)
      library(dplyr)
      
      df <- data.frame(ds = index(df$close[(1):nrow(df)]), y = as.numeric(df$close[(1):nrow(df)]))
      print(head(df))
      #model_prophet <- prophet(df)
      #future <- make_future_dataframe(model_prophet, periods = 30)
      #forecast_prophet <- predict(model_prophet, future)
      
      
      
      
      mean_arima<-as.numeric(forecast_arima$mean)
      Forecast.Prediction<-ifelse(mean_arima>mean(df1$close),"Up",ifelse(mean_arima<mean(df1$close),"Down","No change"))
      min_forecast=round(min(as.numeric(forecast_arima$lower[,2]),na.rm=TRUE),2)
      max_forecast=round(max(as.numeric(forecast_arima$upper[,2]),na.rm=TRUE),2)
      
      latest_price=round(get_latest_price_alpaca(symbol),2)

      df1<-df1%>%mutate(Date=gsub(rownames(df1),pattern="[\\w|\\W]* ",replacement="",perl=TRUE))%>%mutate(Date=as.POSIXct(Date, format = "%H:%M:%S"))
      df1<-df1%>%filter(Date>as.POSIXct("09:30:00", format = "%H:%M:%S"))
      
      p1=ggplot(df1,aes(x = Date, y = close))+geom_line(size = 1.2) +geom_point(aes(color=Rating))+labs(title = paste0(symbol," Stock Price Chart\n","Latest price: $",latest_price,"    ","Price Forecast ($): ",min_forecast," - ",max_forecast),
                                                                                                                                    x = "Time", y = "Price") +
        theme_minimal()+scale_x_datetime(
          breaks = seq(min(df1$Date), max(df1$Date), by = "30 min"),
          date_labels = "%H:%M"
        )+scale_color_manual(values = c("BUY" = "orange", "HOLD" = "black"))
      print(ggplotly(p1))
    })
    
    # Create the ratings table
    output$ratingTable <- renderTable({
      
     # Calculate ratings
      ratings <- df1%>%select(Symbol,Score,Rating)
      
      ratings_long<-df2%>%select(Symbol,Score,Rating)
      data.frame(Stock = ratings$Symbol[nrow(ratings)],
                 #`Latest Buy Score`=ratings$Score[nrow(ratings)],
                 `Rating.last_price` = ratings$Rating[nrow(ratings)],
                 `Rating.previous_close` = ratings_long$Rating[nrow(ratings_long)]
                 )
    })
    
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
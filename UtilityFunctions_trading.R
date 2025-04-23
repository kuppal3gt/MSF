
library(rvest)
library(httr)
library(jsonlite)
library(rvest)
library(quantmod)
library(plyr)
library(prophet)
library(dplyr)
library(SentimentAnalysis)
library(PerformanceAnalytics)
library(forecast)
library(tibble)
library(matrixStats)
library(quantmod)
library(TTR)
library(xgboost)
library(randomForest)
library(caret)
library(PerformanceAnalytics)

library(jsonlite)
library(rvest)
library(quantmod)
library(plyr)
library(prophet)
library(dplyr)
library(SentimentAnalysis)
library(PerformanceAnalytics)
library(forecast)
library(tibble)
library(matrixStats)
library(quantmod)
library(TTR)
library(xgboost)
library(randomForest)
library(caret)
library(PerformanceAnalytics)



if(FALSE){
  allquotes<-getQuote(SYMs$Symbol,what=yahooQF(c("Market Cap (Real-time)","Last Trade (Real-time) With Time", "Change Percent (Real-time)", 
                                                 "Last Trade Size", "Change From 52-week High", "Percent Change From 52-week High", 
                                                 "Last Trade (With Time)", "Last Trade (Price Only)", "High Limit", 
                                                 "Low Limit", "Days Range","Days Range (Real-time)", "50-day Moving Average", 
                                                 "200-day Moving Average", "Change From 200-day Moving Average", 
                                                 "Percent Change From 200-day Moving Average", "Change From 50-day Moving Average", 
                                                 "Percent Change From 50-day Moving Average", "Name", "Notes", 
                                                 "Open", "Previous Close", "Price Paid", "Change in Percent", 
                                                 "Price/Sales", "Price/Book", "Ex-Dividend Date", "P/E Ratio", 
                                                 "Dividend Pay Date", "P/E Ratio (Real-time)", 
                                                 "PEG Ratio", "Price/EPS Estimate Current Year", 
                                                 "Price/EPS Estimate Next Year", "Symbol", "Earnings Timestamp")))
  
  write.csv(allquotes,file="C:\\Users\\karan\\OneDrive\\Documents\\Productivity\\mystockforecast\\allquotes_yahoo.csv",row.names=FALSE)
  
}

stockquotes<-read.csv("allquotes_yahoo.csv") #"C:\\Users\\karan\\OneDrive\\Documents\\Productivity\\mystockforecast\\allquotes_yahoo.csv")



optnames <- c("Ask", "Average Daily Volume", "Ask Size", "Bid", "Ask (Real-time)", 
              "Bid (Real-time)", "Book Value", "Bid Size", "Change & Percent Change", 
              "Change", "Commission", "Change (Real-time)", "After Hours Change (Real-time)", 
              "Dividend/Share", "Last Trade Date", "Trade Date", "Earnings/Share", 
              "Error Indication (returned for symbol changed / invalid)", "EPS Forward","EPS Current Year", 
              "EPS Estimate Next Year", "EPS Estimate Next Quarter", "Float Shares", 
              "Days Low","Days High", "52-week Low", "52-week High", "Holdings Gain Percent", 
              "Annualized Gain", "Holdings Gain", "Holdings Gain Percent (Real-time)", 
              "Holdings Gain (Real-time)", "More Info", "Order Book (Real-time)", 
              "Market Capitalization", "Market Cap (Real-time)", "EBITDA", 
              "Change From 52-week Low", "Percent Change From 52-week Low", 
              "Last Trade (Real-time) With Time", "Change Percent (Real-time)", 
              "Last Trade Size", "Change From 52-week High", "Percent Change From 52-week High", 
              "Last Trade (With Time)", "Last Trade (Price Only)", "High Limit", 
              "Low Limit", "Days Range","Days Range (Real-time)", "50-day Moving Average", 
              "200-day Moving Average", "Change From 200-day Moving Average", 
              "Percent Change From 200-day Moving Average", "Change From 50-day Moving Average", 
              "Percent Change From 50-day Moving Average", "Name", "Notes", 
              "Open", "Previous Close", "Price Paid", "Change in Percent", 
              "Price/Sales", "Price/Book", "Ex-Dividend Date", "P/E Ratio", 
              "Dividend Pay Date", "P/E Ratio (Real-time)", 
              "PEG Ratio", "Price/EPS Estimate Current Year", 
              "Price/EPS Estimate Next Year", "Symbol", "Shares Owned", "Short Ratio", 
              "Last Trade Time", "Trade Links", "Ticker Trend", "1 yr Target Price", 
              "Volume", "Holdings Value", "Holdings Value (Real-time)", "52-week Range", 
              "Days Value Change", "Days Value Change (Real-time)", "Stock Exchange", 
              "Dividend Yield","Average Analyst Rating","Year-to-Date Return","YTD Return","ytdReturn","Price Hint","Net Assets","Earnings Timestamp","Shares Outstanding","Div Yield","Dividend Rate")



dict_words<-c("increase in revenue","revenue grew","demonstrated|showed revenue growth","Strong performance","FDA approval","fold increase","Reduction in Operating Loss","year-over-year increase","growth in subscription revenue","EPS increased","Reports Record EPS","Reports Record sales","revenue growth","Quarterly Gross Margins Increased","Market authorization","FDA Approved","Market authorization","Fast track designation","Priority review","Positive advisory committee vote","Regulatory approval","Commercial launch","Positive results","Successful trial","Significant improvement","efficacy demonstrated","Primary endpoint met","Promising data","Earnings beat","increase in shareholders equity","signs a memorandum of understanding","increase in year-over-year revenue","revenue up","Revenue grew","to develop a","% revenue growth to","increase in gross profit","issued", "dosing begins","announces phase 1 clinical trial initiation","announces successful","successfully completes","outperforms","gross profits","FDA approves","beat expectations","top estimates","tops revenue","beats expectations","tops estimates","surges","grants","granted","announced financial results","announces financial results","Net Profit","Receives Fast Track Designation","Granted Additional Patent protection","favorable safety profile","Improved Efficacy","revenue Surges","continued crowth","Debt Elimination","FDA Grants Priority Review","Received Fast Track Designation","Granted Fast Track Designation","met primary endpoint","meets primary endpoint","Announces Positive Phase 3 Data","demonstrated significant clinical","effectively reduce","a new patent","announces promising","announces expansion","reports promising","reports positive","announce positive","announces positive","announced positive","Participants Showed Improvement","clinically significant improvements","patient benefit","data validates","announces FDA clearance","significant benefit","new positive data","FDA Final Approval","(successful outcome|positive results|improved efficacy|better efficacy|favorable outcome|clear benefit|efficacy demonstrated|statistically significant improvement|significant improvement|clinically meaningful|clinical benefit|robust response|well tolerated|patient benefit|patient improvement|improvement in quality of life|better survival|overall survival improvement|increased survival rate|improved health outcomes|reduction in symptoms|reduced risk|lowered mortality|favorable review|positive feedback|approval recommendation|approved by the agency|support from regulatory bodies|meeting regulatory expectations|regulatory approval|successful meeting|priority review|accelerated approval|breakthrough therapy designation|positive advisory committee feedback|positive CHMP opinion|improved response rates|effective treatment|greater response|achieved primary endpoint|milestone reached|met the endpoints|outcome achieved)","(increased earnings|earnings growth|higher earnings|earnings beat expectations|earnings exceeded expectations|earnings outperform|positive earnings|strong earnings|better-than-expected earnings|profit increase|growth in profits|strong financial results|revenue growth|higher revenue|revenue increase|record earnings|record revenue|earnings rise|profit rise|growing earnings|sales growth|increased sales|strong sales|higher sales|better sales|record sales|sales outperform|sales exceeded expectations|positive sales growth|strong revenue|higher-than-expected revenue|revenue outperform|sales rise|growing sales|sales surge|improved revenue|positive cash flow|FDA Nod|FDA clearance|Eyes Regulatory Filings|approval|Grants Label Expansion|trial met all endpoints|trial meets endpoints|Receives Groundbreaking|510(k) Clearance|NASA Awards|Inducement Grants|receives FDA go-ahead|Non-Brokered Private Placement|Year-Over-Year Growth|Tops Revenue Estimates|Revenue Surge|Revenue Increase)")

dict_words<-c(tolower(dict_words),c("revenue [\\w|\\W]* increased","profit [\\w|\\W]* increased","sales [\\w|\\W]* increased","profits [\\w|\\W]* increased","gains [\\w|\\W]* increased","met all milestones","capital gains","revenues [\\w|\\W]* increased","granted [\\w|\\W]* approval", "encouraging [\\w|\\W]* data",
                                    "promising [\\w|\\W]* data","positive [\\w|\\W]* data","fda [\\w|\\W]* fast track designation", "report positive [\\w|\\W]* data", "announces [\\w|\\W]* approval to","demonstrated [\\w|\\W]* risk reduction","receive [\\w|\\W]* fda [\\w|\\W]*approval","receives positive [\\w|\\W]* opinions","presents [\\w|\\W]* long-term extension data","confirm [\\w|\\W]* safety [\\w|\\W]* profile","become a preferred therapy","encouraging [\\w|\\W]* improvement","demonstrates encouraging [\\w|\\W]* data","announces issuance of [\\w|\\W]* patent","announces [promising|successful|positive]+ [\\w|\\W]*meeting [\\w|\\W]*fda","demonstrates [\\w|\\W]*benefit*[\\w|\\W]*patient","report [\\w|\\W]*positive","[0-9]{1,}% [\\w|\\W]* risk reduction","fold [\\w|\\W]* risk reduction","fda [\\w|\\W]* approval","fda [\\w|\\W]* approved","awards [\\w|\\W]* contract","[announces|reports|announced|reported|declared|declares]+ [\\w|\\W]* inducement grants under nasdaq listing rule 5635","doses first [subject|patient]+","initiation of [\\w|\\W]* phase [\\w|\\W]* trial","initiation of phase [\\w|\\W]* trial","increases [\\w|\\W]* revenue","beat [\\w|\\W]* expectations","receives FDA"))

dict_words<-unique(dict_words)

evaluate_performance<-function(df){
  
  
  df2<-df%>%dplyr::group_by(symbol,side)%>%dplyr::summarise(totalqty=sum(qty,na.rm=TRUE), avgval=mean(price,na.rm=TRUE))
  
  df3<-df2%>%tidyr::pivot_wider(id_cols = c("symbol"),names_from = c("side"),values_from =c("totalqty","avgval" ))
  
  df4<-df3%>%mutate(qty_diff=totalqty_buy-totalqty_sell)
  
  df5<-df4%>%filter(qty_diff==0)%>%mutate(ProfitLoss=(totalqty_buy*avgval_sell-totalqty_buy*avgval_buy),ProfitLoss_Pct=get_pct_diff(avgval_buy,avgval_sell))
  
  #m2=f1$dfraw%>%filter(symbol=="ADTX" & side=="sell")%>%select(price)
  
  return(list(dfraw=df,dfsum=df4,dfmatch=df5,net_profit=sum(df5$ProfitLoss),net_profit_pct=sum(df5$ProfitLoss_Pct)))
  
  
}
get_stock_news<-function(){
  
  ## News from PRNewswire
  web_link<-"https://www.prnewswire.com/news-releases/financial-services-latest-news/financial-services-latest-news-list/?page=1&pagesize=500"
  h1=rvest::read_html(web_link)
  paragraph<- h1 %>% html_nodes(".row") %>% html_text2()
  paragraph<-paragraph[stringr::str_detect(string=paragraph,pattern="(NYSE:)|(NASDAQ:)")==TRUE]%>%strsplit(split="\\n")
  paragraph<-unique(unlist(paragraph))
  paragraph<-paragraph[stringr::str_detect(string=paragraph,pattern="(NYSE:)|(NASDAQ:)")==TRUE]
  
  sym_name<-stringr::str_extract(paragraph,"(NYSE:|NASDAQ:)(\\s)*[A-Z]{2,}")
  
  m2<-data.frame(Symbol=sym_name,Text=paragraph)
  
  m2$Symbol<-gsub(m2$Symbol,pattern="[\\w|\\W]*:",replacement="",perl=TRUE)
  
  #sentiment analysis polarity
  polarity<-analyzeSentiment(paste0(m2$Text))$SentimentQDAP
  polarity<-ifelse(polarity<=0.03,0,0.5)
  
  #score based on pattern matching
  pattern_score<-analyzeDict(dict_words,text = paste0(m2$Text))
  
  ##assign a score based on sentiment analysis and pattern search
  Sentiment<-polarity+pattern_score
  
  Sentiment<-round(Sentiment,3)
  
  m2<-cbind(m2,Sentiment)
  
  watchlist_PN<-m2$Symbol[which(m2$Sentiment>=1)]
  
  watchlist_PN<-gsub(watchlist_PN,pattern="[\\s]*",replacement="",perl=TRUE)
  
  m2B<-m2
  
  m2PR<-m2
  
  m2B$Text=paste0(m2$Text)
  
 # m2PR$Symbol<-paste0("<a href='","https://finance.yahoo.com/quote/",m2$Symbol,"/news/","' target='_blank'>",m2$Symbol,"</a>")
  
  
  web_link<-paste("https://www.globenewswire.com/en/search/exchange/Nasdaq,NYSE/subject/fin,ipo,coa,reg,prs,pdt,ana,shr,anr,cli,fil,ccw,cra,awd,div,ern,emi,jvn,lic,msa,mrr,mna,nav,pat,prt,rgi,res,rcn,tea/date/24HOURS/lang/en?pageSize=50",sep="")
  
  h1=rvest::read_html(web_link)
  Sys.sleep(1)
  web_link2<-"https://www.globenewswire.com/en/search/tag/biotech,artificial%252520intelligence,earnings,fda%252520approval,patents/date/24HOURS?pageSize=50"
  h2=rvest::read_html(web_link2)
  Sys.sleep(1)
  web_link3<-"https://www.globenewswire.com/en/search/date/24HOURS/industry/Pharmaceuticals%2520&%2520Biotechnology?pageSize=50"
  h3=rvest::read_html(web_link3)
  Sys.sleep(1)
  web_link4<-"https://www.globenewswire.com/en/search/date/24HOURS/industry/Technology/lang/en?pageSize=50"
  h4=rvest::read_html(web_link4)
  
  web_link0<-"https://www.globenewswire.com/en/search/lang/en/exchange/Nasdaq,NYSE,AMEX?pageSize=50"
  h0=rvest::read_html(web_link0)
  
  article_category0<- h0 %>% html_nodes(".pagging-list-item-text-container") %>% html_text2()
  
  
  #html_nodes(".lister__header")html_nodes(".lister__article-date")html_nodes("[itemprop='description']")
  article_category<- h1 %>% html_nodes(".pagging-list-item-text-container") %>% html_text2()
  
  article_category2<-h2 %>% html_nodes(".pagging-list-item-text-container") %>% html_text2()
  
  article_category3<-h3 %>% html_nodes(".pagging-list-item-text-container") %>% html_text2()
  
  article_category4<-h4 %>% html_nodes(".pagging-list-item-text-container") %>% html_text2()
  
  web_link5<-"https://www.globenewswire.com/en/search/exchange/Nasdaq/keyword/awards/country/us/date/24HOURS?pageSize=50"
  h5=try(rvest::read_html(web_link5),silent=TRUE)
  
  if(is(h5,"try-error")){
    
    article_category5<-""
    article_category_cols<-strsplit(unique(c(article_category0,article_category,article_category2,article_category3,article_category4)),split = "\\r")
    
  }else{
    
    article_category5<-h5 %>% html_nodes(".pagging-list-item-text-container") %>% html_text2()
    article_category_cols<-strsplit(unique(c(article_category0,article_category,article_category2,article_category3,article_category4,article_category5)),split = "\\r")
    
  }
  
  
  m1<-lapply(1:length(article_category_cols),function(i){cbind(article_category_cols[[i]][4],article_category_cols[[i]][6],article_category_cols[[i]][9],article_category_cols[[i]][13])})
  
  
  m2<-ldply(m1,rbind)
  colnames(m2)<-c("Title","Date","Company","Text")
  
  #sentiment analysis polarity
  polarity<-analyzeSentiment(paste0(m2$Title," ",m2$Text))$SentimentQDAP
  polarity<-ifelse(polarity<=0.03,0,0.5)
  
  #score based on pattern matching
  pattern_score<-analyzeDict(dict_words,text = paste0(m2$Title," ",m2$Text))
  
  ##assign a score based on sentiment analysis and pattern search
  Sentiment<-polarity+pattern_score
  
  Sentiment<-round(Sentiment,3)
  
  m2<-cbind(m2,Sentiment)
  
  t1<-SYMs #TTR::stockSymbols(c("NASDAQ"))
  t1$Name<-gsub(t1$Name,pattern=" - Ordinary Shares| - Common Stock| - Class A Common Stock | - Unit | - Warrant | Daily ETF |  - Class B Ordinary Shares |  - Warrant | - Common Stock",replacement="",ignore.case = T)
  
  t1$Name<-gsub(t1$Name,pattern="[,|.|-]",replacement="",ignore.case = T)
  m2_symbol<-lapply(1:nrow(m2),function(i){
    
    stock_name<-tolower(gsub(m2$Company[i],pattern="^[\\s]*",replacement="",perl = T))
    stock_name<-gsub(stock_name,pattern=" \\(DBA Lifeward\\)",replacement="",ignore.case = T)
    stock_name<-gsub(stock_name,pattern="[,|.|-]",replacement="",ignore.case = T)
    #print(stock_name)
    regex_check<-stringr::str_extract(tolower(t1$Name),tolower(stock_name)) 
    sym_name<-t1$Symbol[which(is.na(regex_check)==FALSE)[1]]
    if(is.na(sym_name)==TRUE){
      
      sym_name<-stringr::str_extract(m2$Text[i],"NASDAQ: [A-Z]{3,}")
      sym_name<-gsub(sym_name,pattern="NASDAQ: ",replacement="")
    }
    return(sym_name)
  })
  m2_symbol<-unlist(m2_symbol)
  
  m2$Symbol<-m2_symbol
  
  
  
  #m2<-m2%>%mutate(Polarity=ifelse(Sentiment>=0.03,1,ifelse(Sentiment<(-0.03),-1,0)))
  
  #m2$Date<-as.Date(m2$Date,format="%m/%d/%Y")
  
  #only select symbols that match the dictionary terms for positive news
  watchlist_GN<-m2_symbol[which(m2$Sentiment>=0.5)]
  
  m2$Text=paste0(m2$Title," ",m2$Text)
  
  m2A<-m2
  m2GN<-m2
  
  
  #m2GN$Symbol<-paste0("<a href='","https://finance.yahoo.com/quote/",m2$Symbol,"/news/","' target='_blank'>",m2$Symbol,"</a>")
  
  library(rvest)
  
  web_link<-paste("https://seekingalpha.com/market-news/")
  
  h1=try(rvest::read_html(web_link),silent=TRUE)
  if(is(h1,"try-error")){
    text1<-""
  }else{
    text1<- h1 %>% html_nodes("article") %>% html_text2()
  }
  
  Sys.sleep(1)
  
  web_link<-paste("https://seekingalpha.com/market-news/earnings")
  
  h2=try(rvest::read_html(web_link),silent=TRUE)
  
  if(is(h2,"try-error")){
    
  }else{
    text1<- rbind(text1,h2 %>% html_nodes("article") %>% html_text2())
  }
  
  Sys.sleep(1)
  
  web_link<-paste("https://seekingalpha.com/market-news/on-the-move")
  
  h3=try(rvest::read_html(web_link),silent=TRUE)
  
  if(is(h3,"try-error")){
    
  }else{
    text1<- rbind(text1,h3 %>% html_nodes("article") %>% html_text2())
  }
  
  Sys.sleep(1)
  
  web_link<-paste("https://seekingalpha.com/market-news/technology")
  
  h4=try(rvest::read_html(web_link),silent=TRUE)
  
  if(is(h4,"try-error")){
    
  }else{
    
    text1<- rbind(text1,h4 %>% html_nodes("article") %>% html_text2())
  }
  Sys.sleep(1)
  
  web_link<-paste("https://seekingalpha.com/market-news/biotech-stocks")
  
  h5=try(rvest::read_html(web_link),silent=TRUE)
  
  if(is(h5,"try-error")){
    
  }else{
    text1<- rbind(text1,h5 %>% html_nodes("article") %>% html_text2())
  }
  
  Sys.sleep(1)
  
  web_link<-paste("https://seekingalpha.com/market-news/energy")
  
  h6=try(rvest::read_html(web_link),silent=TRUE)
  
  if(is(h6,"try-error")){
    
  }else{
    text1<- rbind(text1,h6 %>% html_nodes("article") %>% html_text2())
  }
  if(length(text1)>0){
    text1<-ldply(text1,rbind)
    
    text1<-text1[stringr::str_detect(string=text1[,1],pattern="Today|Yesterday")==TRUE,]
    
    text3=strsplit(text1,"\n")
    
    text4=lapply(1:length(text3),function(i){
      
      news_text=text3[[i]][[1]]
      stock_sym=text3[[i]][[2]]
      sym_name<-stringr::str_extract(stock_sym,"[A-Z]{2,5} [+|-|0]*")
      sym_name<-gsub(sym_name,pattern=" [+|-|0]*",replacement="")
      return(list(news_text=news_text,sym_name=sym_name))
    })
    
    
    text5=t(sapply(text4,as.data.frame))
    
    m2sa=as.data.frame(text5)
    colnames(m2sa)=c("Text","Symbol")
    #text5=gsub(text5,pattern="News\n",replacement="")
    #sentiment analysis polarity
    polarity<-analyzeSentiment(paste0(m2sa$Text))$SentimentQDAP
    polarity<-ifelse(polarity<=0.03,0,0.5)
    
    #score based on pattern matching
    pattern_score<-analyzeDict(dict_words,text = paste0(m2sa$Text))
    
    ##assign a score based on sentiment analysis and pattern search
    Sentiment<-polarity+pattern_score
    
    Sentiment<-round(Sentiment,3)
    
    m2sa<-cbind(m2sa,Sentiment)
    
    
    watchlist_SA<-unique(unlist(m2sa$Symbol[which(m2sa$Sentiment>=1)]))
    
    m2sa2<-m2sa
    
   # m2sa2$Symbol<-paste0("<a href='","https://finance.yahoo.com/quote/",m2sa$Symbol,"/news/","' target='_blank'>",m2sa2$Symbol,"</a>")
    #DT::datatable(m2sa2%>% unique(),filter="top",escape = FALSE,rownames = FALSE)
  }else{
    
    watchlist_SA<-NA
    m2sa<-data.frame(Text=NA,Symbol=NA,Sentiment=NA)
  }
  
  news_analysis<-rbind(m2GN%>%select(Symbol,Sentiment,Text),m2sa2%>%select(Symbol,Sentiment,Text),m2PR%>%select(Symbol,Sentiment,Text))
  DT::datatable(news_analysis%>% unique()%>%arrange(desc(Sentiment)),filter="top",escape=FALSE,options=list(scrollX=TRUE,scrollY=TRUE))
  
  return(news_analysis)
}
get_market_health<-function(symbol){
  
  premarket_data<-get_stock_data_days(symbol, days=200,daily=TRUE) #get_stockdata_alpaca(symbol,"1minute")
  if(length(premarket_data)>0){
  premarket_data<-premarket_data%>%rowwise()%>%mutate(pct_diff1=get_pct_diff(low,high),pct_diff2=get_pct_diff(open,close))
  p1=tail(premarket_data,10) #>%filter(timestamp>paste0(Sys.Date()," 8:00:00") & timestamp<paste0(Sys.Date()," 9:30:00"))
  
  prev_mean<-mean(p1$close)
  pct_diff1_mean<-mean(p1$pct_diff1)
  pct_diff2_mean<-mean(p1$pct_diff2)
  last_price<-get_latest_price_alpaca(symbol)
  pct_diff_last_mean<-get_pct_diff(prev_mean,last_price)
  
  #p1<-rbind(p1,p1[nrow(p1),])
  #p1[nrow(p1),"vwap"]<-get_latest_price_alpaca(symbol)
  #print(p1)
  #print(prev_mean)
  #print(last_price)
  #c1<-cor(seq(1,nrow(p1)),p1$vwap)
  #print(c1)
  #c2<-cor(p1$open,p1$close)
  #print("C2")
  #print(c2)
  #print(pct_diff1_mean)
  #print(pct_diff2_mean)
  #print(pct_diff_last_mean) #c2>0.5 & 
  premarket_status<-ifelse(pct_diff_last_mean>(0.1) & c1>0.5 & last_price>prev_mean & pct_diff1_mean<0.5 & pct_diff2_mean>(-0.3),1,
                           ifelse(pct_diff_last_mean<(0.1) & c1<0.5 & last_price<prev_mean & pct_diff1_mean<0.5 & pct_diff2_mean>(-0.3),-1,ifelse(c1<0 | c2<0,-1,0)))
  return(premarket_status)
  }else{
    return(0)
  }
}


#function for pattern matching based on dictionary of keywords and regular expressions
analyzeDict<-function(dict_words,text){
  
  res<-lapply(1:length(text),function(i){
    length(which(stringr::str_detect(string=tolower(text[i]),pattern=dict_words))==TRUE)
  })
  res<-unlist(res)
  return(res)
}



get_daily_trending_symbols<-function(){
  
  
  url_link<-"https://query1.finance.yahoo.com/v1/finance/trending/US?count=50&amp;fields=logoUrl%2ClongName%2CshortName%2CregularMarketChange%2CregularMarketChangePercent%2CregularMarketPrice&amp;format=true&amp;useQuotes=true&amp;quoteType=ALL&amp;lang=en-US&amp;region=US"
  
  d1<-fromJSON(url_link)
  
  watchlist1<-d1$finance$result$quotes[[1]]
  watchlist1<-as.character(watchlist1$symbol)
  return(unique(watchlist1))
}

get_daily_active_symbols<-function(){
  
  ActiveTicker<-read_html("https://www.google.com/finance/markets/most-active?hl=en")
  watch_list<-ActiveTicker%>%html_nodes(".COaKTb")%>%html_text2()

  # Trending Tickers from Google
  if(length(which(watch_list=="Index")[1])>0){
    
    watch_list<-watch_list[-grep(x=watch_list,pattern="Index")] 
  }
  
  if(length(which(watch_list%in%c("CORZZ","VSTE","GV","Index")))>0){
    watch_list<-watch_list[-which(watch_list%in%c("CORZZ","VSTE","GV","Index"))]
  }
  watch_list<-na.omit(watch_list)
  
  return(unique(watch_list))
}

get_daily_losers_symbols<-function(){
  
  ActiveTicker<-read_html("https://www.google.com/finance/markets/losers?hl=en")
  watch_list<-ActiveTicker%>%html_nodes(".COaKTb")%>%html_text2()
  
  # Trending Tickers from Google
  if(length(which(watch_list=="Index")[1])>0){
    
    watch_list<-watch_list[-grep(x=watch_list,pattern="Index")] 
  }
  
  if(length(which(watch_list%in%c("CORZZ","VSTE","GV","Index")))>0){
    watch_list<-watch_list[-which(watch_list%in%c("CORZZ","VSTE","GV","Index"))]
  }
  watch_list<-na.omit(watch_list)
  return(unique(watch_list))
}

get_daily_gainers_symbols<-function(){
  
  ActiveTicker<-read_html("https://www.google.com/finance/markets/gainers?hl=en")
  watch_list<-ActiveTicker%>%html_nodes(".COaKTb")%>%html_text2()
  
  # Trending Tickers from Google
  if(length(which(watch_list=="Index")[1])>0){
    
    watch_list<-watch_list[-grep(x=watch_list,pattern="Index")] 
  }
  
  if(length(which(watch_list%in%c("CORZZ","VSTE","GV","Index")))>0){
    watch_list<-watch_list[-which(watch_list%in%c("CORZZ","VSTE","GV","Index"))]
  }
  watch_list<-na.omit(watch_list)
  return(unique(watch_list))
}

get_daily_movers_symbols<-function(){
  

    url_link<-"https://query1.finance.yahoo.com/v1/finance/trending/US?count=50&amp;fields=logoUrl%2ClongName%2CshortName%2CregularMarketChange%2CregularMarketChangePercent%2CregularMarketPrice&amp;format=true&amp;useQuotes=true&amp;quoteType=ALL&amp;lang=en-US&amp;region=US"
    
    d1<-fromJSON(url_link)
    
    watchlist1<-d1$finance$result$quotes[[1]]
    watchlist1<-as.character(watchlist1$symbol)
    ActiveTicker<-read_html("https://www.google.com/finance/markets/most-active?hl=en")
    watchlist3<-ActiveTicker%>%html_nodes(".COaKTb")%>%html_text2()
    #print(watchlist3)
    # Trending Tickers from Google
    if(length(which(watchlist3=="Index")[1])>0){
      
      watchlist3<-watchlist3[-grep(x=watchlist3,pattern="Index")] #watchlist3[1:(which(watchlist3=="Index")[1]-1)]
    }
    
    ActiveTicker2<-read_html("https://www.google.com/finance/markets/losers?hl=en")
    watchlist4<-ActiveTicker2%>%html_nodes(".COaKTb")%>%html_text2()
    
    if(length(which(watchlist4=="Index")[1])>0){
      
      watchlist4<-watchlist4[-grep(x=watchlist4,pattern="Index")] #watchlist4[1:(which(watchlist4=="Index")[1]-1)]
    }
    
    ActiveTicker3<-read_html("https://www.google.com/finance/markets/gainers?hl=en")
    watchlist5<-ActiveTicker3%>%html_nodes(".COaKTb")%>%html_text2()
    
    if(length(which(watchlist5=="Index")[1])>0){
      
      watchlist5<-watchlist5[-grep(x=watchlist5,pattern="Index")] #watchlist5[1:(which(watchlist5=="Index")[1]-1)]
    }
    
    # watchlist6=get_premarket_gainers()$symbol
    #watchlist1,watchlist2,,watchlist6,watchlist6
    watch_list<-c(watchlist1,watchlist3,watchlist4,watchlist5)
    
    delta_thresh=2
    watch_list<-unique(watch_list)
    
    if(length(which(watch_list%in%c("CORZZ","VSTE","GV","Index")))>0){
      watch_list<-watch_list[-which(watch_list%in%c("CORZZ","VSTE","GV","Index"))]
    }
    watch_list<-na.omit(watch_list)
    
    #watch_list<-c(watch_list[which(watch_list%in%SYMs$Symbol)],custom_watchlist)
    
 
  
  return(watch_list)
}

get_daily_movers<-function(SYMs,custom_watchlist=c("SOXS","SOXL","CWEB"),check.categ="all"){
  
  if(check.categ=="all"){
  url_link<-"https://query1.finance.yahoo.com/v1/finance/trending/US?count=50&amp;fields=logoUrl%2ClongName%2CshortName%2CregularMarketChange%2CregularMarketChangePercent%2CregularMarketPrice&amp;format=true&amp;useQuotes=true&amp;quoteType=ALL&amp;lang=en-US&amp;region=US"
  
  d1<-fromJSON(url_link)
  
  watchlist1<-d1$finance$result$quotes[[1]]
  watchlist1<-as.character(watchlist1$symbol)
  ActiveTicker<-read_html("https://www.google.com/finance/markets/most-active?hl=en")
  watchlist3<-ActiveTicker%>%html_nodes(".COaKTb")%>%html_text2()
  print(watchlist3)
  # Trending Tickers from Google
  if(length(which(watchlist3=="Index")[1])>0){
    
    watchlist3<-watchlist3[-grep(x=watchlist3,pattern="Index")] #watchlist3[1:(which(watchlist3=="Index")[1]-1)]
  }
  
  ActiveTicker2<-read_html("https://www.google.com/finance/markets/losers?hl=en")
  watchlist4<-ActiveTicker2%>%html_nodes(".COaKTb")%>%html_text2()
  
  if(length(which(watchlist4=="Index")[1])>0){
    
    watchlist4<-watchlist4[-grep(x=watchlist4,pattern="Index")] #watchlist4[1:(which(watchlist4=="Index")[1]-1)]
  }
  
  ActiveTicker3<-read_html("https://www.google.com/finance/markets/gainers?hl=en")
  watchlist5<-ActiveTicker3%>%html_nodes(".COaKTb")%>%html_text2()
  
  if(length(which(watchlist5=="Index")[1])>0){
    
    watchlist5<-watchlist5[-grep(x=watchlist5,pattern="Index")] #watchlist5[1:(which(watchlist5=="Index")[1]-1)]
  }
  
 # watchlist6=get_premarket_gainers()$symbol
  #watchlist1,watchlist2,,watchlist6,watchlist6
  watch_list<-c(watchlist1,watchlist3,watchlist4,watchlist5)
  
  watch_list<-c(watchlist4,watchlist5)
  
  delta_thresh=2
  watch_list<-unique(watch_list)
  
  if(length(which(watch_list%in%c("CORZZ","VSTE","GV","RBRK")))>0){
    watch_list<-watch_list[-which(watch_list%in%c("CORZZ","VSTE","GV","RBRK"))]
  }
  watch_list<-na.omit(watch_list)
  
  watch_list<-c(watch_list[which(watch_list%in%SYMs$Symbol)],custom_watchlist) #
  print(watch_list)
  }else{
    
    watch_list<-custom_watchlist
    
  }
  
  watch_list<-unique(watch_list)
  stockres_final1<-get_stock_analysis(watch_list=watch_list,delta_thresh=0.5,period_val=3)
  
  #fname<-paste0("stockres_final1_Google",Sys.Date(),".Rda")
  
  #save(stockres_final1,file=fname)
  
  
  #stockres_final1$Symbol<-paste0("<a href='","https://finance.yahoo.com/quote/",stockres_final1$Symbol,"","' target='_blank'>",stockres_final1$Symbol,"</a>")
  
  #stockres_final1$Trade.Signal<-as.factor(stockres_final1$Trade.Signal)
  #stockres_final3b<-stockres_final1%>%dplyr::rename(R.SP=Cor.withSP500,Rating=Trade.Signal,`Buy Range`=Buy.Range)%>%mutate(Price.EPS=round(Price.EPS,2),fastD=as.numeric(as.character(fastD)),OnFire=ifelse(Volume>1.5*`Ave. Daily Volume` & Volume>SMA5_volume,"High",ifelse(Volume>1.5*`Ave. Daily Volume` | Volume>SMA5_volume,"Medium","Low")))%>%select(Symbol,MarketCapCat,`Forecast Range`,`Forecast Trend`,Rating,R.SP,`Buy Range`,OnFire,ytd_return,Last,P.Close,Low,High,Ask,Bid,`EPS Forward`,`EPS Current Year`,`%Change`,`20 day MA`,`50 day MA`,`5 day MA`,fastK,fastD,`Ave. Analyst Rating`,SMA5_volume,`Ave. Daily Volume`,Volume,`52 week Low`,`52 week High`,Name,daily_mean,weekly_mean,monthly_mean,daily_sd,weekly_sd,monthly_sd,sharpe_ratio,var_95)
  
  #,Forecast,Cor.withSP500 ,Forecast,Cor.withSP500
  #DT::datatable(stockres_final3b,caption = Sys.Date(),width = "1800px",filter ="top",options=list(scrollX = TRUE,scrollY=TRUE,autoWidth = FALSE,columnDefs=list(list(width="10px",targets="_all"))),escape = FALSE,rownames = FALSE)%>%DT::formatStyle(columns = seq(1,ncol(stockres_final3b)), fontSize = '100%')
  
  
  #mat3<-merge(stockquotes%>%select(Name,Symbol,Earnings.Timestamp),mat3,by="Symbol")
  
  mat3<-stockres_final1
  
  mat3$Symbol<-paste0("<a href='","https://finance.yahoo.com/quote/",mat3$Symbol,"","' target='_blank'>",mat3$Symbol,"</a>")
  
  
  DT::datatable(mat3,caption = Sys.Date(),width = "1800px",filter ="top",options=list(scrollX = TRUE,scrollY=TRUE,search = list(regex = TRUE, caseInsensitive = TRUE),autoWidth = FALSE,columnDefs=list(list(width="10px",targets="_all"))),escape = FALSE,rownames = FALSE)%>%DT::formatStyle(columns = seq(1,ncol(stockres_final1)), fontSize = '100%')
  
  return(stockres_final1)
}

get_symbols_daily_gainers<-function(){
  
  #SYMs <- TTR::stockSymbols(c("NASDAQ","NYSE"))
  #SYMs$Name[which(SYMs$Symbol=="IDAI")]<-c("Trust Stamp Inc. - Class A Common Stock")
  
  #& Exchange%in%c("NASDAQ","NYSE") 
  #SYMs<-SYMs%>%as.data.frame()%>%filter(Test.Issue==FALSE & ETF==FALSE & Financial.Status!="Deficient")
  allquotes<-update_stock_quotes()
  allquotes<-allquotes%>%filter(CapCategory%in%c("Mid","Large"))
  
  allquotes=allquotes%>%filter(`% Change`>(3)) #&  grepl(x=`Ave. Analyst Rating`,pattern="Buy"))
  
  allquotes<-allquotes[order(allquotes$`% Change`,decreasing=TRUE),]
  
  w3<-allquotes$Symbol
  #get_daily_losers_symbols()
  w3=w3[1:c(min(50,length(w3)))]
  
  watch_list<-unique(c(w3)) 
  #,"AAPL","MSFT","AMZN","NVDA","GOOGL","META","LLY","AVGO","JPM","UNH","JNJ","V","XOM","PG","MA","F","CSCO","ADBE","ORCL","ACN","TMO","ABT","BMY","AMGN",
  #                     "MCD","SBUX","BKNG","TJX","GM","KO","PEP","WMT","BAC","WFC","GS","CVX","COP","SLB","EOG","PLTR","AMT","PLD","CCI","EQIX","PPG","LIN"))
  #watch_list<-watch_list[which(watch_list%in%SYMs$Symbol)]
  SYMs<-allquotes%>%select(Symbol,Name,CapCategory)
  #news_symbols,watch_list2res$Symbol
  stockres_final3b<-get_daily_movers(SYMs,unique(watch_list),check.categ = "custom")
  #select_symbols<-unique(c(watch_list$Symbol,watch_list2res$Symbol,get_daily_movers_symbols(),unique(custom_watchlist),news_symbols))
  #stockres_final3a<-get_daily_movers(SYMs,unique(c(watch_list)),check.categ = "custom")
  #s
  #select symbols that are above pivot point and 50day MA


  active_table_select1<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceUp"))>0,1,0))%>%filter(`Risk(%)`>=(-5) & `%Change`>(3) 
                                                                                                                            & buy_signal==1 & Buy.Rating==1 
                                                                                                                            & `Popularity Level`=="High" & Price.EPS>0 & `Year-to-Date Return`>0 & MarketCap%in%c("Mid","Large"))
  
  
  active_table_select2<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceUp"))>0,1,0))%>%filter(`Risk(%)`>=(-10) & `%Change`>(5) 
                                                                                                                           & RSI.Check==1 & Buy.Rating==1 & sell_signal==1
                                                                                                                           & Price.EPS>0  & MarketCap%in%c("Mid","Large") & Pivot.Check>0 & MA5.Check>0
                                                                                                                           & MA20.Check>0 & MA50.Check>0)
  
  active_table_select<-rbind(active_table_select1,active_table_select2)
  
  
  
 
  active_table_select<-active_table_select[order(active_table_select$`%Change`),] #%>%filter(`%Change`<(2))
  
  active_table_select<-active_table_select%>%unique()
  
  select_symbols<-active_table_select$Symbol%>%unique()
  
  select_symbols<-select_symbols[which(select_symbols%in%stockres_final3b$Symbol[which(stockres_final3b$MarketCap%in%c("Mid","Large"))])]
  
  return(list(select_symbols=select_symbols,stockres=stockres_final3b))
}

get_symbols_daily_losers<-function(){
  
  #SYMs <- TTR::stockSymbols(c("NASDAQ","NYSE"))
  #SYMs$Name[which(SYMs$Symbol=="IDAI")]<-c("Trust Stamp Inc. - Class A Common Stock")
  
  #& Exchange%in%c("NASDAQ","NYSE") 
  #SYMs<-SYMs%>%as.data.frame()%>%filter(Test.Issue==FALSE & ETF==FALSE & Financial.Status!="Deficient")
  allquotes<-update_stock_quotes()
  save(allquotes,file="allquotes.Rda")
  
  allquotes<-allquotes%>%filter(CapCategory%in%c("Mid","Large"))
  
  allquotes=allquotes%>%filter(`% Change`<(-3) &  grepl(x=`Ave. Analyst Rating`,pattern="Buy"))
  
  allquotes<-allquotes[order(allquotes$`% Change`),]
  
  w3<-allquotes$Symbol
  #get_daily_losers_symbols()
  w3=w3[1:c(min(50,length(w3)))]
  
  watch_list<-unique(c(w3,"AAPL","MSFT","AMZN","NVDA","GOOGL","META","LLY","AVGO","JPM","UNH","JNJ","V","XOM","PG","MA","F","CSCO","ADBE","ORCL","ACN","TMO","ABT","BMY","AMGN",
                       "MCD","SBUX","BKNG","TJX","GM","KO","PEP","WMT","BAC","WFC","GS","CVX","COP","SLB","EOG","PLTR","AMT","PLD","CCI","EQIX","PPG","LIN"))
  #watch_list<-watch_list[which(watch_list%in%SYMs$Symbol)]
  SYMs<-allquotes%>%select(Symbol,Name,CapCategory)
  #news_symbols,watch_list2res$Symbol
  stockres_final3b<-get_daily_movers(SYMs,unique(watch_list),check.categ = "custom")
  #select_symbols<-unique(c(watch_list$Symbol,watch_list2res$Symbol,get_daily_movers_symbols(),unique(custom_watchlist),news_symbols))
  #stockres_final3a<-get_daily_movers(SYMs,unique(c(watch_list)),check.categ = "custom")
  #s
  #select symbols that are above pivot point and 50day MA
  active_table_select1<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="Buy: PriceDownBelowMA"))>0,1,ifelse(length(grep(Rating,pattern="TrendDown"))>0,-1,0)))%>%filter(`Year-to-Date Return`>3 & `%Change`<(2) & `Risk(%)`>=(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & MA50.Check<0 & Price.Status=="Oversold" & Buy.Rating>=0 & R.SP>0.6)
  
  active_table_select1B<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="Buy: PriceDownBelowMA"))>0,1,0))%>%filter(`Year-to-Date Return`>10 & `%Change`<(1) & `Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & fastDKratio>=1.2 & Buy.Rating==1 & R.SP>0.5 & Prev.Check<0 & MA50.Check<0 & Price.Status=="Oversold")
  
  # & `20 day MA`>`50 day MA` & `20 day MA`>`5 day MA`
  
  active_table_select1C<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceUp"))>0,1,0))%>%filter(`Year-to-Date Return`>10 & `%Change`<(1) & `Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & R.SP>0.8 & Prev.Check>0 &  Buy.Rating==1 & MA50.Check<0 & Price.Status=="Oversold")
  
  active_table_select1D<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Year-to-Date Return`>0 & `%Change`<(1) & `Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & fastDKratio>=1.1 & buy_signal==1 & R.SP>0.5 & Prev.Check<0 & MA50.Check<0 & `20 day MA`>`50 day MA` & `20 day MA`>`5 day MA` & MA50.Check<0 & MA5.Check<0 & Buy.Rating==1)
  
  active_table_select1E<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Year-to-Date Return`>0 & `%Change`<(1) & `Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & fastDKratio>=1.1 & buy_signal==1 & R.SP>0.5 & Prev.Check>0 & MA50.Check<0 & MA50.Check<0 & MA5.Check<0 & Buy.Rating==1)
  
  active_table_select1F<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`%Change`>(0) & `Risk(%)`>=(-5) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & fastDKratio<=0.75 & buy_signal==1 & Prev.Check>0 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & Price.Status=="Oversold")
  
  active_table_select1G<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & buy_signal==1 & Prev.Check>0 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & Price.Status%in%c("Oversold","Overbought"))
  
  active_table_select1H<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `Year-to-Date Return`>(-3) & `%Change`>(-15) & buy_signal==1 & Prev.Check>0 & RSI.Check==1 & Pivot.Check<0 & buy_signal==1)
  
  active_table_select1I<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & buy_signal==1 & Prev.Check>0 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & fastDKratio>1.2 & Last>Low)
  
  active_table_select1J<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(-15) & buy_signal==1 & `5 day MA`>`20 day MA` & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & fastDKratio>1.2 & Last>Low)
  
  active_table_select1K<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceUp"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(5) & buy_signal==1 & Buy.Rating==1 & `Popularity Level`=="High" & `Year-to-Date Return`>0)
  
  active_table_select1L<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Year-to-Date Return`>0 & `Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & buy_signal==1 & Prev.Check>0 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0)
  
  active_table_select1M<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & buy_signal==1 & Prev.Check<0 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & Price.Status%in%c("Oversold","Overbought"))
  
  active_table_select1N<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Year-to-Date Return`>0 & `Risk(%)`>=(-8) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & buy_signal==1 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0)
  
  active_table_select1O<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceUp"))>0,1,0))%>%filter(`Year-to-Date Return`>0 & `Risk(%)`>=(-8) & `%Change`>(-15) & MA5.Check>0 & MA20.Check>0 & MA50.Check>0 & Pivot.Check>0 & Price.Status%in%c("Overbought"))
  
  active_table_select1P<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & Price.Status%in%c("Oversold"))
  
  active_table_select1Q<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-10) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & sell_signal==1 & Prev.Check>0)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1B)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1C)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1D)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1E)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1F)
  
  
  active_table_select1<-rbind(active_table_select1,active_table_select1G)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1H)
  
  
  active_table_select1<-rbind(active_table_select1,active_table_select1I)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1J)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1K)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1L)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1M)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1N)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1O)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1P)
  active_table_select1<-rbind(active_table_select1,active_table_select1Q)
  
  active_table_select2<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="Buy"))>0,1,0))%>%filter(`Year-to-Date Return`>10 & fastDKratio>1 & buy_signal==1 & `%Change`<1 & Pivot.Low.Check>0 & RSI>50 & RSI<71 & WilliamR<(-80) & Pivot.Check==1 & MA5.Check<0 & MA50.Check>0 & RSI.Check>0 & sell_signal==0)
  
  #stockres_final3b%>%filter(`Year-to-Date Return`>10 & fastDKratio>1 & buy_signal==1 & Pivot.Low.Check>0 & Prev.Check>0 & MA50.Check<0 & `%Change`<0 & `Risk(%)`>=(-15) & MarketCap%in%c("Mid","Large"))
  
  active_table_select3<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="Buy"))>0,1,0))%>%filter(Symbol%in%c("SOXL","SOXS","CWEB") & `%Change`<(-3) & buy_signal==1)
  #active_table_select3<-stockres_final3b%>%filter(`Year-to-Date Return`>10 & Pivot.Check>0 & MA5.Check<0 & MA50.Check>0 & `%Change`<0 & `Risk(%)`>=(-5) & Price.Status=="Oversold" & MarketCap%in%c("Mid","Large"))
  
  
  active_table_select<-rbind(active_table_select1,active_table_select2)
  
  active_table_select<-rbind(active_table_select,active_table_select3)
  
  active_table_select<-active_table_select[order(active_table_select$`%Change`),] #%>%filter(`%Change`<(2))
  
  active_table_select<-active_table_select%>%unique()
  
  select_symbols<-active_table_select$Symbol%>%unique()
  
  select_symbols<-select_symbols[which(select_symbols%in%stockres_final3b$Symbol[which(stockres_final3b$MarketCap%in%c("Mid","Large"))])]
  
  return(list(select_symbols=select_symbols,stockres=stockres_final3b))
}

get_symbols_daily_movers<-function(){
  
  SYMs <- TTR::stockSymbols(c("NASDAQ","NYSE"))
  SYMs$Name[which(SYMs$Symbol=="IDAI")]<-c("Trust Stamp Inc. - Class A Common Stock")
  
  #& Exchange%in%c("NASDAQ","NYSE") 
  SYMs<-SYMs%>%as.data.frame()%>%filter(Test.Issue==FALSE & ETF==FALSE & Financial.Status!="Deficient")
  
  
  watch_list2<-get_marketmovers()%>%filter(abs(as.numeric(as.character(percent_change)))>1) #%>%rowwise()%>%
  watch_list2res<-getQuote(watch_list2$symbol)
  watch_list2res<-watch_list2res%>%filter(abs(`% Change`)>3 & Volume>1000000)
  watch_list2res<-cbind(rownames(watch_list2res),watch_list2res)
  cnames1<-colnames(watch_list2res)
  cnames1[1]<-"Symbol"
  colnames(watch_list2res)<-cnames1
  
  #n1<-get_stock_news()
  #n1<-n1%>%as.data.frame()
  #news_symbols<-na.omit(unlist(n1$Symbol[which(n1$Sentiment>1)]))
  
  #largecap<-read.csv("C:\\Users\\karan\\OneDrive\\Documents\\Productivity\\mystockforecast\\LargeCap_by_Sector.csv",allowEscapes = TRUE,as.is = TRUE)
  
  #largecap<-read.csv("C:\\Users\\karan\\OneDrive\\Documents\\Productivity\\mystockforecast\\allquotes_yahoo.csv")
  #largecap<-largecap%>%filter(CapCategory%in%c("Large Cap") & !Symbol%in%c("HEI-A","SQ") & as.numeric(as.character(Last))<250
  #                            & is.na(Ex.Dividend.Date)==FALSE & grepl(Ave..Analyst.Rating,pattern = "Buy"))
  
  #cnames_f<-colnames(largecap)
  #cnames_f[1]<-c("Symbol")
  
  #largecap<-largecap%>%filter(!Sector%in%c("Technology") & as.numeric(as.character(PE.Ratio..TTM.))>50)
 # largecap<-largecap%>%as.data.frame()%>%dplyr::filter(as.numeric(as.character(PE.Ratio..TTM.))>50 & as.numeric(as.character(Price..Intraday.))<250 & !Symbol%in%c("HEI-A","SQ"))
  #watch_list<-c(largecap$Symbol) #gsub(sp500_list$Symbol,pattern=" ",replacement="")
  
  w1=get_daily_active_symbols()
  w1=w1[1:c(min(20,length(w1)))]
  w2=get_daily_trending_symbols()
  w2=w1[2:c(min(20,length(w1)))]
  w3=get_daily_losers_symbols()
  w3=w3[1:c(min(25,length(w3)))]
  w4=get_daily_gainers_symbols()
  w4=w4[1:c(min(25,length(w4)))]
  
  watch_list<-unique(c(w1,w2,w3,w4,watch_list2res$Symbol,"SOXS","SOXL","CWEB","NVDX","SPMO","INTC","AAPL","MSFT","AMZN","NVDA","GOOGL","META","LLY",
                       "AVGO","JPM","UNH","JNJ","V","XOM","PG","MA","F","CSCO","ADBE","ORCL","ACN","TMO","ABT","BMY","AMGN",
                       "MCD","SBUX","BKNG","TJX","GM","KO","PEP","WMT","BAC","WFC","GS","CVX","COP","SLB","EOG","PLTR","AMT","PLD","CCI","EQIX","PPG","LIN"))
  watch_list<-watch_list[which(watch_list%in%SYMs$Symbol)]
  #news_symbols,watch_list2res$Symbol
  stockres_final3b<-get_daily_movers(SYMs,unique(watch_list),check.categ = "custom")
  #select_symbols<-unique(c(watch_list$Symbol,watch_list2res$Symbol,get_daily_movers_symbols(),unique(custom_watchlist),news_symbols))
  #stockres_final3a<-get_daily_movers(SYMs,unique(c(watch_list)),check.categ = "custom")
  #s
  #select symbols that are above pivot point and 50day MA
  active_table_select1<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="Buy: PriceDownBelowMA"))>0,1,ifelse(length(grep(Rating,pattern="TrendDown"))>0,-1,0)))%>%filter(`Year-to-Date Return`>3 & `%Change`<(2) & `Risk(%)`>=(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & MA50.Check<0 & Price.Status=="Oversold" & Buy.Rating>=0 & R.SP>0.6)
  
  active_table_select1B<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="Buy: PriceDownBelowMA"))>0,1,0))%>%filter(`Year-to-Date Return`>10 & `%Change`<(1) & `Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & fastDKratio>=1.2 & Buy.Rating==1 & R.SP>0.5 & Prev.Check<0 & MA50.Check<0 & Price.Status=="Oversold")
  
  # & `20 day MA`>`50 day MA` & `20 day MA`>`5 day MA`
  
  active_table_select1C<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceUp"))>0,1,0))%>%filter(`Year-to-Date Return`>10 & `%Change`<(1) & `Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & R.SP>0.8 & Prev.Check>0 &  Buy.Rating==1 & MA50.Check<0 & Price.Status=="Oversold")
  
  active_table_select1D<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Year-to-Date Return`>0 & `%Change`<(1) & `Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & fastDKratio>=1.1 & buy_signal==1 & R.SP>0.5 & Prev.Check<0 & MA50.Check<0 & `20 day MA`>`50 day MA` & `20 day MA`>`5 day MA` & MA50.Check<0 & MA5.Check<0 & Buy.Rating==1)
  
  active_table_select1E<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Year-to-Date Return`>0 & `%Change`<(1) & `Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & fastDKratio>=1.1 & buy_signal==1 & R.SP>0.5 & Prev.Check>0 & MA50.Check<0 & MA50.Check<0 & MA5.Check<0 & Buy.Rating==1)
  
  active_table_select1F<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`%Change`>(0) & `Risk(%)`>=(-5) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & fastDKratio<=0.75 & buy_signal==1 & Prev.Check>0 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & Price.Status=="Oversold")
  
  active_table_select1G<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & buy_signal==1 & Prev.Check>0 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & Price.Status%in%c("Oversold","Overbought"))
  
  active_table_select1H<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `Year-to-Date Return`>(-3) & `%Change`>(-15) & buy_signal==1 & Prev.Check>0 & RSI.Check==1 & Pivot.Check<0 & buy_signal==1)
  
  active_table_select1I<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & buy_signal==1 & Prev.Check>0 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & fastDKratio>1.2 & Last>Low)
  
  active_table_select1J<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(-15) & buy_signal==1 & `5 day MA`>`20 day MA` & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & fastDKratio>1.2 & Last>Low)
  
  active_table_select1K<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceUp"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(5) & buy_signal==1 & Buy.Rating==1 & `Popularity Level`=="High" & `Year-to-Date Return`>0)
  
  active_table_select1L<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Year-to-Date Return`>0 & `Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & buy_signal==1 & Prev.Check>0 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0)
  
  active_table_select1M<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Risk(%)`>=(-15) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & buy_signal==1 & Prev.Check<0 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0 & Price.Status%in%c("Oversold","Overbought"))
  
  active_table_select1N<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceDownBelowMA"))>0,1,0))%>%filter(`Year-to-Date Return`>0 & `Risk(%)`>=(-8) & `%Change`>(-15) & MarketCap%in%c("Mid","Large") & Price.EPS>0 & buy_signal==1 & MA5.Check<0 & MA20.Check<0 & MA50.Check<0 & Pivot.Check<0)
  
  active_table_select1O<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="PriceUp"))>0,1,0))%>%filter(`Year-to-Date Return`>0 & `Risk(%)`>=(-8) & `%Change`>(-15) & MA5.Check>0 & MA20.Check>0 & MA50.Check>0 & Pivot.Check>0 & Price.Status%in%c("Overbought"))
  
  
  active_table_select1<-rbind(active_table_select1,active_table_select1B)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1C)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1D)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1E)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1F)
  
  
  active_table_select1<-rbind(active_table_select1,active_table_select1G)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1H)
  
  
  active_table_select1<-rbind(active_table_select1,active_table_select1I)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1J)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1K)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1L)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1M)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1N)
  
  active_table_select1<-rbind(active_table_select1,active_table_select1O)
  
  active_table_select2<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="Buy"))>0,1,0))%>%filter(`Year-to-Date Return`>10 & fastDKratio>1 & buy_signal==1 & `%Change`<1 & Pivot.Low.Check>0 & RSI>50 & RSI<71 & WilliamR<(-80) & Pivot.Check==1 & MA5.Check<0 & MA50.Check>0 & RSI.Check>0 & sell_signal==0)
  
  #stockres_final3b%>%filter(`Year-to-Date Return`>10 & fastDKratio>1 & buy_signal==1 & Pivot.Low.Check>0 & Prev.Check>0 & MA50.Check<0 & `%Change`<0 & `Risk(%)`>=(-15) & MarketCap%in%c("Mid","Large"))
  
  active_table_select3<-stockres_final3b%>%mutate(Buy.Rating=ifelse(length(grep(Rating,pattern="Buy"))>0,1,0))%>%filter(Symbol%in%c("SOXL","SOXS","CWEB") & `%Change`<(-3) & buy_signal==1)
  #active_table_select3<-stockres_final3b%>%filter(`Year-to-Date Return`>10 & Pivot.Check>0 & MA5.Check<0 & MA50.Check>0 & `%Change`<0 & `Risk(%)`>=(-5) & Price.Status=="Oversold" & MarketCap%in%c("Mid","Large"))
  
  
  active_table_select<-rbind(active_table_select1,active_table_select2)
  
  active_table_select<-rbind(active_table_select,active_table_select3)
  
  active_table_select<-active_table_select[order(active_table_select$`%Change`),] #%>%filter(`%Change`<(2))
  
  active_table_select<-active_table_select%>%unique()
if(FALSE){
  stock.health<-lapply(active_table_select$Symbol,function(symbol){
    res<-try(evaluate_stock_health(symbol),silent=TRUE)
    if(is(res,"try-error")){
      res<-FALSE
    }
    return(res)
    })
  #stock.health<-unlist(stock.health)
  
  active_table_select<-active_table_select%>%mutate(stock.health=as.data.frame(stock.health)) 
  
  #active_table_select<-active_table_select%>%filter(stock.health==TRUE)%>%as.data.frame()
  
  #active_table_select<-active_table_select%>%unique()%>%rowwise()%>%mutate(current.health=get_market_health(symbol=Symbol))
  trade.signal<-lapply(active_table_select$Symbol,function(symbol){
    res<-try(get_trade_signal_daily(symbol),silent=TRUE)
    if(is(res,"try-error")){
      res<-"HOLD"
    }
    return(res)
    
  
  })
  #trade.signal<-unlist(trade.signal)
  
  active_table_select<-active_table_select%>%mutate(trade.signal=as.data.frame(trade.signal))
}
  select_symbols<-active_table_select$Symbol%>%unique()
  
  return(list(select_symbols=select_symbols,stockres=stockres_final3b))
}

# Categorize function with all five tiers
categorize_cap <- function(market_cap) {
  
  if(is.na(market_cap)==FALSE){
  
  if (market_cap >= 1e10) {
    return("Large")
  } else if (market_cap >= 2e9) {
    return("Mid")
  } else if (market_cap >= 3e8) {
    return("Small")
  } else if (market_cap >= 5e7) {
    return("Micro")
  } else {
    return("Nano")
  }
  }else{
    
    return(NA)
  }
}

update_stock_quotes<-function(){
  
  SYMs <- TTR::stockSymbols(c("NASDAQ","NYSE"))
  SYMs$Name[which(SYMs$Symbol=="IDAI")]<-c("Trust Stamp Inc. - Class A Common Stock")
  #setwd("C:\\Users\\karan\\OneDrive\\Documents\\Productivity\\mystockforecast\\")
  #& Exchange%in%c("NASDAQ","NYSE") 
  SYMs<-SYMs%>%as.data.frame()%>%mutate(Financial.Status=ifelse(is.na(Financial.Status)==TRUE,"Unknown",Financial.Status))%>%filter(Test.Issue==FALSE & ETF==FALSE & Financial.Status!="Deficient")
  
  if(FALSE){
    allquotes<-getQuote(SYMs$Symbol,what=yahooQF(c("Market Capitalization","Last Trade (Price Only)", "Change in Percent", 
                                                   "Change From 52-week High", "Percent Change From 52-week High", 
                                                   "Last Trade (With Time)", "Last Trade (Price Only)", "Days High", 
                                                   "Days Low", "Days Range","Days Range (Real-time)", "50-day Moving Average", 
                                                   "200-day Moving Average", "Change From 200-day Moving Average", 
                                                   "Percent Change From 200-day Moving Average", "Change From 50-day Moving Average", 
                                                   "Percent Change From 50-day Moving Average", "Name",  
                                                   "Open", "Previous Close", "Price Paid", "Change in Percent", 
                                                   "Price/Sales", "Price/Book", "Ex-Dividend Date", "P/E Ratio", 
                                                   "Dividend Pay Date", "P/E Ratio (Real-time)", 
                                                   "PEG Ratio", "Price/EPS Estimate Current Year", 
                                                   "Price/EPS Estimate Next Year", "Symbol", "Earnings Timestamp","Shares Outstanding",
                                                   "EPS Forward","EPS Current Year","Year-to-Date Return","Average Analyst Rating","Net Assets"
                                                    )))
  }
    allquotes<-getQuote(SYMs$Symbol,what=yahooQF(c("Name","50-day Moving Average","Market Capitalization","Dividend Pay Date",
                                                   "Ex-Dividend Date", "P/E Ratio","Earnings Timestamp",
                                                   "EPS Forward","Average Analyst Rating","Change in Percent")))
    allquotes<-allquotes%>%mutate(Symbol=rownames(allquotes))
    
    # Apply the function
    allquotes$CapCategory <- sapply(allquotes$`Market Capitalization`, categorize_cap)
    
    #write.csv(allquotes,file="~/Downloads/allquotes_yahoo.csv",row.names=FALSE)
    return(allquotes)
}

get_previous_day_losers<-function(){
  
  allquotes<-update_stock_quotes()
  
  allquotes<-allquotes%>%filter(CapCategory%in%c("Mid","Large"))
  symbols<-allquotes$Symbol
  
  #get_stock_data_window("AAPL",Sys.Date()-1,Sys.Date(),"Day")
  
}

pred_stock_trend<-function(stock_symbol,r1)
{
  if(FALSE){
    Sys.sleep(0.1)
    # Get stock data
    stock_data<-getSymbols(stock_symbol, src = "yahoo", from = "2024-01-01", to = Sys.Date(),auto.assign = FALSE)
    
    colnames(stock_data)<-c("Open","High","Low","Close","Volume","Adjusted")
    
    AAPL<-stock_data
    
    # Calculate technical indicators
    current_price <- Cl(AAPL)
    previous_close <- Lag(Cl(AAPL), 1)
    change_in_volume <- Vo(AAPL) - Lag(Vo(AAPL), 1)
    avg_volume <- runMean(Vo(AAPL), n = 50)
    volume_ratio <- avg_volume / Vo(AAPL)
    rsi <- RSI(Cl(AAPL), n = 14)
    stochastic_k <- stoch(Cl(AAPL), nFastK = 14, nFastD = 3)$fastK
    stochastic_d <- stoch(Cl(AAPL), nFastK = 14, nFastD = 3)$fastD
    bid_price <- (Hi(AAPL) + Lo(AAPL)) / 2
    ask_price <- (Hi(AAPL) + Lo(AAPL)) / 2
    ma_20 <- SMA(Cl(AAPL), n = 20)
    ma_5 <- SMA(Cl(AAPL), n = 5)
    ma_50 <- SMA(Cl(AAPL), n = 50)
    momentum <- momentum(Cl(AAPL), n = 3)
    
    CCI_res <- try(TTR::CCI(na.omit(stock_data[,c("High","Low","Close")])),silent=TRUE)
    WPR_res <- try(TTR::WPR(na.omit(stock_data[,c("High","Low","Close")])),silent=TRUE) 
    
    macd <- MACD(Cl(AAPL), nFast = 12, nSlow = 26, nSig = 9)$macd
    bollinger_bands <- BBands(Cl(AAPL), n = 20)
    atr <- ATR(HLC(AAPL), n = 14)$atr
    #momentum <- momentum(Cl(AAPL), n = 10)
    obv <- OBV(Cl(AAPL), Vo(AAPL))
    
    # Combine indicators into a data frame
    indicators <- data.frame(
      current_price = as.numeric(current_price),
      previous_close = as.numeric(previous_close),
      change_in_volume = as.numeric(change_in_volume),
      # volume_ratio = as.numeric(volume_ratio),
      rsi = as.numeric(rsi),
      stochastic_k = as.numeric(stochastic_k),
      #  stochastic_d = as.numeric(stochastic_d),
      # stochastic_kd=stochastic_k/stochastic_d,
      #bid_price = as.numeric(bid_price),
      # ask_price = as.numeric(ask_price),
      ma_20 = as.numeric(ma_20),
      ma_5 = as.numeric(ma_5),
      ma_50 = as.numeric(ma_50),
      momentum = as.numeric(momentum)
      #   macd=as.numeric(macd),
      #bollinger_bands=as.numeric(bollinger_bands),
      # atr=as.numeric(atr),
      #  obv=as.numeric(obv)
      
      #cci_res=as.numeric(CCI_res),
      #wpr_res=as.numeric(WPR_res)
    )
    
    indicators<-indicators%>%mutate(ma_20=ifelse(is.na(ma_20)==TRUE,ma_5,ma_20),ma_50=ifelse(is.na(ma_50)==TRUE,ma_5,ma_50))
    
    # Create target variable for up or down trend over the next 3 trading days
    indicators$trend <- indicators$trend <- ifelse(lead(Cl(AAPL), 3) > Cl(AAPL) & lead(Cl(AAPL), 2) > Cl(AAPL) & lead(Cl(AAPL), 1) > Cl(AAPL), 1, 0)
    
    indicators<-na.omit(indicators)
    # Split data into training and testing sets
    set.seed(123)
    train_index <- createDataPartition(indicators$trend, p = 0.8, list = FALSE)
    train_data <- indicators[train_index, ]
    test_data <- indicators[-train_index, ]
    
    train_data<-as.data.frame(train_data)
    test_data<-as.data.frame(test_data)
    
    # Define control for cross-validation
    train_control <- trainControl(method = "cv", number = 5)
    if(FALSE){        
      # Train XGBoost model with hyperparameter optimization
      xgb_grid <- expand.grid(
        nrounds = c(100, 500),
        max_depth = c(3, 10),
        eta = c(0.001, 0.01),
        gamma = c(0, 1),
        colsample_bytree = c(0.5, 0.8),
        min_child_weight = c(1, 5),
        subsample = c(0.5, 0.8)
      )
      xgb_model <- caret::train(
        as.factor(trend) ~ .,
        data =train_data,
        method = "xgbTree",
        trControl = train_control,
        tuneGrid = xgb_grid,metric="Accuracy"
      )
    }
    train_ctrl <- trainControl(method="cv", # type of resampling in this case Cross-Validated
                               number=5, # number of folds
                               search = "grid" # we are performing a "grid-search"
    )
    
    grid_rf <- expand.grid( mtry=c(1:20) )
    
    # Train Random Forest model with hyperparameter optimization
    rf_grid <- expand.grid(
      mtry = seq(2, 6),
      splitrule = c("variance"),
      min.node.size = c(1, 5)
      
    )
    #tuneGrid = rf_grid,
    rf_model <-train(as.factor(trend) ~ .,
                     data =train_data,
                     method = "rf", # this will use the randomForest::randomForest function
                     metric = "Accuracy", # which metric should be optimized for 
                     trControl = train_ctrl, 
                     tuneGrid =grid_rf,
                     # options to be passed to randomForest
                     ntree = 741,
                     keep.forest=TRUE,
                     importance=TRUE
    ) 
    
    # Train Support Vector Machine (SVM) model with hyperparameter optimization
    svm_grid <- expand.grid(
      C = c(0.0001,0.001, 0.005, 1,10),
      sigma = c(0.0001,0.001, 0.005, 0.1,1)
    )
    svm_model <- train(
      as.factor(trend) ~ ., data = train_data,
      method = "svmRadial",metric="Accuracy",
      trControl = train_control,
      tuneGrid = svm_grid
    )
    
    
    # Evaluate models on test data
    # xgb_predictions <- predict(xgb_model, newdata = test_data)
    rf_predictions <- predict(rf_model, newdata = test_data)
    svm_predictions <- predict(svm_model, newdata = test_data)
    #lstm_predictions <- predict_classes(lstm_model, array_reshape(test_data_lstm[, -ncol(test_data_lstm)], c(nrow(test_data_lstm), ncol(test_data_lstm) - 1, 1)))
    svm_accuracy <- sum(diag(table(svm_predictions,test_data$trend)))/nrow(test_data) #mean(rf_predictions == test_data$trend)
    
    #  xgb_accuracy <- sum(diag(table(xgb_predictions,test_data$trend)))/nrow(test_data) #mean(xgb_predictions == test_data$trend)
    rf_accuracy <- sum(diag(table(rf_predictions,test_data$trend)))/nrow(test_data) #mean(rf_predictions == test_data$trend)
    #lstm_accuracy <- sum(diag(table(lstm_predictions,test_data$trend)))/nrow(test_data) #mean(rf_predictions == test_data$trend)
    
    #  cat("XGBoost Accuracy:", xgb_accuracy, "\n")
    cat("Random Forest Accuracy:", rf_accuracy, "\n")
    cat("SVM Accuracy:", svm_accuracy, "\n")
    #cat("LSTM Accuracy:", lstm_accuracy, "\n")
    Sys.sleep(0.1)
    #r1<-get_stock_analysis(watch_list=c(stock_symbol),delta_thresh=0.5,period_val=3)
    
    
    fname<-paste(stock_symbol,"trendmod.Rda",sep="")
    #save(rf_model,xgb_model,svm_model,file=fname)
    save(rf_model,svm_model,file=fname)
  }
  
  fname<-paste(stock_symbol,"trendmod.Rda",sep="")
  load(fname)
  test1<-data.frame(current_price=r1$Last,previous_close=r1$P.Close,change_in_volume=as.numeric(as.character(r1$Volume))-as.numeric(as.character(r1$SMA5_volume)),rsi=r1$RSI,stochastic_k=r1$fastK,bid_price=r1$Bid,ask_price=r1$Ask,ma_20=r1$`20 day MA`,ma_5=r1$`5 day MA`,ma_50=r1$`50 day MA`,momentum=r1$`momentum`)
  
  
  colnames(test1)<-c("current_price","previous_close","change_in_volume","rsi","stochastic_k","ma_20","ma_5","ma_50","momentum") 
  
  #colnames(test_data[,-c(ncol(test_data))])
  trend_pred1<-predict(rf_model, newdata = test1)
  #  trend_pred2<-predict(xgb_model, newdata = test1)
  trend_pred3<-predict(svm_model, newdata = test1)
  #trend_pred4<-predict(lstm_model, newdata = t1)
  #  trend_pred<-median(c(as.numeric(as.character(trend_pred1)),as.numeric(as.character(trend_pred2)),as.numeric(as.character(trend_pred3))),na.rm=TRUE)
  trend_pred<-median(c(as.numeric(as.character(trend_pred1)),as.numeric(as.character(trend_pred3))),na.rm=TRUE)
  return(trend_pred)
}

get_postmarket_gainers<-function(){
  
  
  library(rvest)
  library(dplyr)
  
  # URL of the website
  url <- "https://www.tradingview.com/markets/stocks-usa/market-movers-after-hours-gainers/"
  
  # Read the HTML content from the website
  webpage <- read_html(url)
  
  # Extract the table
  premarket_table <- webpage %>%
    html_node("table") %>%
    html_table()
  
  tooltip_values <- webpage %>%
    html_nodes(".apply-common-tooltip") %>%
    html_text()
  
  tooltip_values<-tooltip_values[-c(1:10)]
  symbol=tooltip_values[seq(1,length(tooltip_values),2)]
  name=tooltip_values[seq(2,length(tooltip_values),2)]
  symbols_name<-data.frame(cbind(symbol,name))
  
  # Convert the string to a numeric value
  convert_to_numeric <- function(str_value) {
    multiplier <- 1
    print(str_value)
    if (grepl("M", str_value[1])) {
      multiplier <- 1e6
    } else if (grepl("B", str_value)) {
      multiplier <- 1e9
    } else if (grepl("K", str_value)) {
      multiplier <- 1e3
    }
    
    numeric_value <- as.numeric(gsub("[^0-9\\.]", "", str_value)) * multiplier
    return(numeric_value)
  }
  
  premarket_table<-cbind(symbols_name,premarket_table)
  
  premarket_table<-premarket_table%>%mutate(Price=gsub(Price,pattern="\\ USD",replacement="",perl=TRUE),`Post-market Chg %`=gsub(`Post-market Chg %`,pattern="\\+|%",
                                                                                                                                 replacement="",perl=TRUE),
                                            `Post-market Close`=gsub(`Post-market Close`,pattern=" USD",replacement=""),
                                           )
  
  premarket_table<-premarket_table%>%mutate(Price=as.numeric(as.character(Price)),`Post-market Chg %`=as.numeric(as.character(`Post-market Chg %`)),
                                            `Post-market Close`=as.numeric(as.character(`Post-market Close`)))%>%mutate(Limit.Buy=`Post-market Close`*0.98,
                                                                                                                       Limit.Sell=`Post-market Close`*1.02)
  
  print("here0")
  premarket_table<-premarket_table%>%rowwise()%>%mutate(`Post-market Vol`=convert_to_numeric(`Post-market Vol`)) #%>%filter(symbol%in%SYMs$Symbol)
  
  #premarket_table<-premarket_table%>%filter(`Post-market Chg %`>5 & `Post-market Vol`>100000)
  #
  previous_quote=getQuote(premarket_table$symbol,what=yahooQF(optnames))
  
  print("here1")
  previous_quote<-cbind(rownames(previous_quote),previous_quote)
  
  print("here2")
  cnames1<-colnames(previous_quote)
  cnames1[1]<-"symbol"
  cnames1[5]<-"Volume2"
  colnames(previous_quote)<-cnames1
  previous_quote<-previous_quote%>%dplyr::rename(Previous.Volume=Volume)
  
  premarket_table<-merge(premarket_table,previous_quote,by="symbol")
  
  premarket_table<-premarket_table%>%mutate(Volume.Ratio.Prev=`Post-market Vol`/Previous.Volume,Volume.Ratio.Avg=`Post-market Vol`/`Ave. Daily Volume`)
  
  premarket_table<-premarket_table[,c("symbol","name","Post-market Chg %","Post-market Close","Price","Limit.Buy","Post-market Vol","Volume.Ratio.Prev","Volume.Ratio.Avg","52-week Low","52-week High","200-day MA","50-day MA","% Change From 200-day MA","% Change From 50-day MA")]
  
  premarket_table<-premarket_table%>%mutate(`% Change From 200-day MA`=100*`% Change From 200-day MA`,`% Change From 50-day MA`=100*`% Change From 50-day MA`)
  # Display the table and titles with hrefs
  premarket_table<-premarket_table[order(premarket_table$`Post-market Chg %`,decreasing = TRUE),]
  
  
  premarket_health<-lapply(premarket_table$symbol,get_premarket_health)
  premarket_health<-unlist(premarket_health)
  
  premarket_table<-premarket_table%>%mutate(premarket.health=premarket_health)%>%rowwise()%>%mutate(stock.health=evaluate_stock_health(symbol)) #get_premarket_health(symbol)
  
  #premarket_table_select<-premarket_table%>%filter(`Pre-market Chg %`>5 & Volume.Ratio.Avg>10)
  return(premarket_table)
}

get_premarket_gainers<-function(){
  
  
  library(rvest)
  library(dplyr)
  
  # URL of the website
  url <- "https://www.tradingview.com/markets/stocks-usa/market-movers-pre-market-gainers/"
  
  # Read the HTML content from the website
  webpage <- read_html(url)
  
  # Extract the table
  premarket_table <- webpage %>%
    html_node("table") %>%
    html_table()
  
  tooltip_values <- webpage %>%
    html_nodes(".apply-common-tooltip") %>%
    html_text()
  
  tooltip_values<-tooltip_values[-c(1:11)]
  symbol=tooltip_values[seq(1,length(tooltip_values),2)]
  name=tooltip_values[seq(2,length(tooltip_values),2)]
  symbols_name<-data.frame(cbind(symbol,name))
  
  # Convert the string to a numeric value
  convert_to_numeric <- function(str_value) {
    multiplier <- 1
    #print(str_value)
    if (grepl("M", str_value[1])) {
      multiplier <- 1e6
    } else if (grepl("B", str_value)) {
      multiplier <- 1e9
    } else if (grepl("K", str_value)) {
      multiplier <- 1e3
    }
    
    numeric_value <- as.numeric(gsub("[^0-9\\.]", "", str_value)) * multiplier
    return(numeric_value)
  }
  
  premarket_table<-cbind(symbols_name,premarket_table)
  
  premarket_table<-premarket_table%>%mutate(Price=gsub(Price,pattern="\\ USD",replacement="",perl=TRUE),`Pre-market Chg %`=gsub(`Pre-market Chg %`,pattern="\\+|%",replacement="",perl=TRUE),`Pre-market Close`=gsub(`Pre-market Close`,pattern=" USD",replacement=""),`Pre-market Gap %`=gsub(`Pre-market Gap %`,pattern="\\+|%",replacement="",perl = TRUE))
  
  premarket_table<-premarket_table%>%mutate(Price=as.numeric(as.character(Price)),`Pre-market Chg %`=as.numeric(as.character(`Pre-market Chg %`)),`Pre-market Close`=as.numeric(as.character(`Pre-market Close`)),`Pre-market Gap %`=as.numeric(as.character(`Pre-market Gap %`)))%>%mutate(Limit.Buy=`Pre-market Close`*0.98,Limit.Sell=`Pre-market Close`*1.02)%>%rowwise()%>%mutate(`Pre-market Vol`=convert_to_numeric(`Pre-market Vol`)) #%>%filter(symbol%in%SYMs$Symbol)
  
  premarket_table<-premarket_table%>%filter(`Pre-market Chg %`>5) #& `Pre-market Vol`>100000
  #
  previous_quote=getQuote(premarket_table$symbol,what=yahooQF(optnames))
  
  previous_quote<-cbind(rownames(previous_quote),previous_quote)
  cnames1<-colnames(previous_quote)
  cnames1[1]<-"symbol"
  cnames1[5]<-"Volume2"
  colnames(previous_quote)<-cnames1
  previous_quote<-previous_quote%>%dplyr::rename(Previous.Volume=Volume)
  
  premarket_table<-merge(premarket_table,previous_quote,by="symbol")
  
  premarket_table<-premarket_table%>%mutate(Volume.Ratio.Prev=`Pre-market Vol`/Previous.Volume,Volume.Ratio.Avg=`Pre-market Vol`/`Ave. Daily Volume`)
  
  premarket_table<-premarket_table[,c("symbol","name","Pre-market Chg %","Pre-market Close","Pre-market Gap %","Price","Limit.Buy","Pre-market Vol","Volume.Ratio.Prev","Volume.Ratio.Avg","52-week Low","52-week High","200-day MA","50-day MA","% Change From 200-day MA","% Change From 50-day MA")]
  
  premarket_table<-premarket_table%>%mutate(`% Change From 200-day MA`=100*`% Change From 200-day MA`,`% Change From 50-day MA`=100*`% Change From 50-day MA`)
  
  premarket_health<-lapply(premarket_table$symbol,get_premarket_health)
  premarket_health<-unlist(premarket_health)
  
  stock_health=lapply(premarket_table$symbol,function(x){evaluate_stock_health(x,FALSE)})
  stock_health<-unlist(stock_health)
  
  premarket_table<-premarket_table%>%mutate(premarket.health=premarket_health,stock.health=stock_health) #get_premarket_health(symbol)
  
  # Display the table and titles with hrefs
  premarket_table<-premarket_table[order(premarket_table$`Pre-market Chg %`,decreasing = TRUE),]
  #premarket_table_select<-premarket_table%>%filter(`Pre-market Chg %`>5 & Volume.Ratio.Avg>10)
  return(premarket_table)
}

get_losers_highdividend<-function(){
  
  
  library(rvest)
  library(dplyr)
  
  # URL of the website
  url <- "https://www.tradingview.com/markets/stocks-usa/market-movers-high-dividend/"
  
  # Read the HTML content from the website
  webpage <- read_html(url)
  
  # Extract the table
  premarket_table <- webpage %>%
    html_node("table") %>%
    html_table()
  
  tooltip_values <- webpage %>%
    html_nodes(".apply-common-tooltip") %>%
    html_text()
  
  tooltip_values<-tooltip_values[-c(1:13)]
  symbol=tooltip_values[seq(1,length(tooltip_values),3)]
  name=tooltip_values[seq(2,length(tooltip_values),3)]
  symbols_name<-data.frame(cbind(symbol,name))
  
  # Convert the string to a numeric value
  convert_to_numeric <- function(str_value) {
    multiplier <- 1
    #print(str_value)
    if (grepl("M", str_value[1])) {
      multiplier <- 1e6
    } else if (grepl("B", str_value)) {
      multiplier <- 1e9
    } else if (grepl("K", str_value)) {
      multiplier <- 1e3
    }
    
    numeric_value <- as.numeric(gsub("[^0-9\\.]", "", str_value)) * multiplier
    return(numeric_value)
  }
  
  premarket_table<-cbind(symbols_name,premarket_table)
  
  premarket_table<-premarket_table%>%mutate(Price=gsub(Price,pattern="\\ USD",replacement="",perl=TRUE),`Change %`=gsub(`Change %`,pattern="%",replacement="",perl=TRUE),
                                            `Div yield % (indicated)`=gsub(`Div yield % (indicated)`,pattern="%",replacement="",perl=TRUE))%>%mutate(`Change %`=gsub(`Change %`,pattern="^[.]",replacement="",perl=TRUE))
  
  premarket_table<-premarket_table%>%mutate(Price=as.numeric(as.character(Price)),
                                            `Change %`=as.numeric(as.character(`Change %`)))%>%mutate(Limit.Buy=`Price`*0.98,
                                            Limit.Sell=`Price`*1.02)%>%rowwise()%>%mutate(`Volume`=convert_to_numeric(`Volume`)) #%>%filter(symbol%in%SYMs$Symbol)
  
  #premarket_table<-premarket_table%>%filter(`Change %`<(-3) & `Div yield % (indicated)`>10) # & `Pre-market Vol`>100000)
  premarket_table<-premarket_table%>%mutate(`Change %`=ifelse(is.na(`Change %`)==TRUE,-1,`Change %`))%>%filter(`Change %`<(0) & `Div yield % (indicated)`>10)
  premarket_table<-premarket_table[order(premarket_table$`Change %`,decreasing = FALSE),]
  #premarket_table_select<-premarket_table%>%filter(`Pre-market Chg %`>5 & Volume.Ratio.Avg>10)
  return(premarket_table)
}

forecast_stock_price<-function(df,periods_val=14){
  
  #stock_symbol,periods_val=7*4,windowrange=365
  if(FALSE){
    
    startrange=Sys.Date()-windowrange
    endrange=Sys.Date()
    temp_df<-getSymbols(stock_symbol, from = startrange, to = endrange,warnings = FALSE,auto.assign = FALSE)
    temp_df<-as.data.frame(temp_df)
    df<-cbind(rownames(temp_df),temp_df[,ncol(temp_df)])
    
  }
  df<-as.data.frame(df)
  colnames(df)<-c("ds","y")
  df<-as.data.frame(df)
  #df$ds<-as.Date(as.character(df$ds),format="%m/%d/%Y")
  df$y<-as.numeric(as.character(df$y))
  a1=auto.arima(as.numeric(as.character(df$y))) #,ic="bic")
  f1=forecast(a1,h=periods_val,level=c(95))
  forecast1<-cbind(seq(1,periods_val),f1$mean,f1$lower,f1$upper)
  forecast1<-as.data.frame(forecast1)
  
  #print(head(forecast1))
  if(FALSE){ 
    m <- prophet(df,weekly.seasonality = TRUE,daily.seasonality = TRUE,seasonality.mode = "multiplicative")
    future <- make_future_dataframe(m, periods = periods_val) #,freq = "month")
    forecast1 <- predict(m, future)
    forecast1<-as.data.frame(forecast1)
    forecast1<-subset(forecast1,select = c("ds","yhat","yhat_lower","yhat_upper"))
    
  }
  colnames(forecast1)<-c("period","yhat","yhat_lower","yhat_upper")
  cor1<-try(cor(as.numeric(as.character(forecast1$yhat)),as.numeric(as.character(forecast1$period)),method="spearman",use="pairwise.complete.obs"),silent=TRUE)
  if(is(cor1,"try-error")){
    forecast_trend<-0
  }else{
    if(is.na(cor1)==TRUE){
      
      cor1=0
    }
    
    if(cor1>0.3){
      forecast_trend<-1
    }else{
      if(cor1<(-0.3)){
        forecast_trend<-(-1)
      }else{
        forecast_trend<-0
      }
    } 
  }
  
  forecast_trend<-0
  Forecast.price.range<-paste(round(min(forecast1$yhat),2),round(max(forecast1$yhat),2),sep="-")
  Forecast.trend<-ifelse(forecast_trend>0,"Positive","Negative")
  Forecast.trend<-forecast_trend
  Forecast.trend<-replace(Forecast.trend,which(forecast_trend<(0)),"Negative")
  Forecast.trend<-replace(Forecast.trend,which(forecast_trend>(0)),"Positive")
  Forecast.trend<-replace(Forecast.trend,which(forecast_trend==0),"Neutral")
  #Beta.measure<-res2$beta
  Forecast.next90days<-Forecast.price.range
  Trend.next90days<-forecast_trend
  
  return(list(forecast.period=forecast1,forecast_trend=Trend.next90days,forecast_range=Forecast.next90days))
  
}


# Function to calculate pivot points
calculate_pivot_points <- function(high, low, close) {
  pivot <- (high + low + close) / 3
  resistance1 <- (2 * pivot) - low
  support1 <- (2 * pivot) - high
  resistance2 <- pivot + (high - low)
  support2 <- pivot - (high - low)
  return(list(pivot = pivot, resistance1 = resistance1, support1 = support1, resistance2 = resistance2, support2 = support2))
}

get_resistance_support<-function(stock_data=NA,sym=NA,tperiod=60){
  
  if(is.na(sym[1])==FALSE){
    stock_data<-getSymbols(sym, src = "yahoo", from = "2024-01-01", to = Sys.Date(),auto.assign = FALSE)
  }
  colnames(stock_data)<-c("Open","High","Low","Close","Volume","Adjusted")
  previous_data<-tail(stock_data,tperiod)
  # Initialize a data frame to store pivot points
  pivot_points_df <- data.frame(Date = index(previous_data), Pivot = NA, Resistance1 = NA, Support1 = NA, Resistance2 = NA, Support2 = NA)
  
  # Loop through each day and calculate pivot points
  for (i in 1:nrow(previous_data)) {
    high <- as.numeric(previous_data$High[i])
    low <- as.numeric(previous_data$Low[i])
    close <- as.numeric(previous_data$Close[i])
    pivot_points <- calculate_pivot_points(high, low, close)
    pivot_points_df$Pivot[i] <- pivot_points$pivot
    pivot_points_df$Resistance1[i] <- round(pivot_points$resistance1,2)
    pivot_points_df$Support1[i] <- round(pivot_points$support1,2)
    pivot_points_df$Resistance2[i] <- round(pivot_points$resistance2,2)
    pivot_points_df$Support2[i] <- round(pivot_points$support2,2)
  }
  
  #print(paste0(median(pivot_points_df$Resistance1)," - ",median(pivot_points_df$Resistance2)))
  # print(paste0(median(pivot_points_df$Support2)," - ",median(pivot_points_df$Support1)))
  # print(median(pivot_points_df$Pivot))
  
  med_piv_point<-median(pivot_points_df$Pivot)
  med_piv_point<-round(med_piv_point,2)
  return(med_piv_point)
  
}


get_buy_sell_signal <- function(slast, spclose,spmedian, sphigh,s5MA,s20MA,s50MA,s120MA,sRSI,sTDI,cor_val,cor_fastK,cor_fastD,sSD,sSD2,sfastK,sfastK2,sfastD,sfastD2,sdi,stdi,sPctChangeClose,sPctChangeMA5,sPctChangeMA20,sPctChangeMA50,weights,volumechange,cor_SMA20,smacd,smacdsignal,scci,swpr,bollinger_bands)
{
  
  score <- sum(weights[1] * 100*((slast-spclose)/spclose),
               weights[2] * 100*((slast-spmedian)/spmedian),
               weights[3] * 100*((slast-sphigh)/sphigh),
               weights[4] * 100*((slast-s5MA)/s5MA),
               weights[5] * 100*((slast-s20MA)/s20MA),
               weights[6] * 100*((slast-s50MA)/s50MA),
               weights[7] * 100*((slast-s120MA)/s120MA))
  
  delta_thresh<-weights[8]*1000
  
  #cor_val=max(df$TDI[1:(i-1)])
  buy_or_sell_signal <- "Neutral"
  inflection_point=TRUE
  # print(score)
  #&& df$TDI[i-1] >0  
  
  #print(c(slast, spclose,spmedian, sphigh,s5MA,s20MA,s50MA,s120MA,sRSI,sTDI,cor_val,cor_fastK,sSD,sSD2,sfastK,sfastK2,sfastD,sfastD2,sdi,stdi,weights))
  
  
  #if(is.na(score)==FALSE)score > delta_thresh && 
  #    {  
  if (slast>s5MA && slast>s20MA && slast>s120MA && slast>spmedian && sRSI>50 &&  sTDI ==100) {
    buy_or_sell_signal <- "PriceUpTrendUp"
  } 
  else if ( sTDI ==1  && sRSI<60) 
  {
    #  && score < delta_thresh 
    if((slast<s50MA && slast<s20MA && cor_val>(-0.7) && slast<spmedian && sSD>0 && cor_fastK>0.7 && sfastD<0.20 && slast<s120MA && slast<s5MA && s5MA<s20MA))
      
      
    {
      
      {
        # buy_or_sell_signal<-"Buy"
        
        
        buy_or_sell_signal <- "PriceDownTrendUp"
      }
    }else{
      
      if((cor_val>(0) &&  sRSI<60 && slast<spmedian && sSD>0 && inflection_point==TRUE && slast<s120MA && slast<s5MA && sdi<0 && stdi>0 && sfastD<0.20 && slast<s50MA && cor_fastK>0.7 && ((100*(s5MA-slast)/s5MA)>1) && ((100*(s50MA-slast)/s50MA)>3)))
        
        
      {
        
        
        buy_or_sell_signal <- "PriceDownTrendUp"
      }    else
      {
        #  && slast>s120MA 
        if((slast<s50MA && slast<s20MA && slast<spmedian && sSD>0 && cor_val>(0) && sRSI<35 && inflection_point==TRUE && slast<s5MA && s5MA<s20MA && stdi<0 && sdi<0 && cor_fastK<(-0.7) && volumechange>2))
          
          
        {
          
          #
          
          buy_or_sell_signal <- "PriceDown"
          
        }else{
          
          #low dip
          if((cor_val>(0) && sRSI<60 && sSD>0 && inflection_point==TRUE && slast>s120MA && slast<s5MA && sdi>0 && slast>s50MA && slast<spmedian && sfastD<0.20 && ((100*(s5MA-slast)/s5MA)>1)))
            
            
          {
            
            
            buy_or_sell_signal <- "PriceDown"
          }else{
            
            #steep dipslast<s120MA sdi<0 && 
            if((cor_val>(0) && sRSI<60 && inflection_point==TRUE && slast<spmedian  && slast<s5MA && cor_fastK>(0.7) && sfastD<0.40 && sfastD<0.40 && slast<s50MA && ((100*(s5MA-slast)/s5MA)>1) && ((100*(s50MA-slast)/s50MA)>3)))
              
              
            {
              
              
              buy_or_sell_signal <- "PriceDownTrendUp"
            }else{
              
              #fastK fastD cross-over&& ((100*(s5MA-slast)/s5MA)>1) && ((100*(s50MA-slast)/s50MA)>3)
              if((cor_val>(0) && sRSI<60 && sSD>0 && inflection_point==TRUE && slast<spmedian && cor_fastK<(-0.7) && slast<s5MA && (sfastK<0.1) && sfastD<0.20 && slast<s50MA && slast<s20MA))
                
                
              {
                
                
                buy_or_sell_signal <- "PriceDownTrendDown"
              }else{
                
                #downward short-term && ((100*(s5MA-slast)/s5MA)>1) && ((100*(s50MA-slast)/s50MA)>3)slast<s50MA && 
                if((cor_val>(0) && sRSI<60 && inflection_point==TRUE && slast>spmedian && slast>s5MA && cor_fastK>(0.9) && sfastK<0.20 && slast<s20MA && slast<s50MA && slast>s120MA && s50MA>s120MA && s50MA>s20MA && sSD>0))
                {
                  
                  buy_or_sell_signal <- "PriceDownTrendUp"
                }else{
                  
                  #trend changing upwards
                  if((cor_val>(0) && sRSI<60 && slast>s120MA && sfastK<0.5 &&  inflection_point==TRUE && sTDI<0 && slast>spmedian && (100*abs(slast-s5MA)/s5MA)<5 && slast<s20MA && slast<s50MA && cor_fastK>(0.9) && (sfastK/sfastD)>=0.95))
                  {
                    buy_or_sell_signal <- "TrendChangeUpwards"
                  }
                  else{
                    #score>delta_thresh && 
                    if (slast>spmedian && slast>s5MA && slast>s20MA && slast>s50MA && slast>s120MA && sRSI>50 &&  sfastK>0.9 && sfastD>0.8 && cor_fastK>(0.8) && (sfastK/sfastD)>1) {
                      buy_or_sell_signal <- "Sell"
                    } else{
                      
                      if((cor_val>(0) && sRSI<60 && inflection_point==TRUE && slast>spmedian && (100*abs(slast-s5MA)/s5MA)<5 && cor_fastK<(-0.7) && sfastK<0.50 && slast<s20MA && slast<s50MA && stdi<0 && sSD<1 && (sfastD/sfastK)>1))
                      {
                        
                        buy_or_sell_signal <- "TrendChangeUpwards"
                      }else{
                        # svolchg>3  && stdi>0 && sdi>0 
                        if((cor_val>(0) && sRSI<80 && sSD>0 && sfastK<0.5 && inflection_point==TRUE && slast>s120MA && slast>s5MA && slast>s50MA && slast>spmedian && (sfastD/sfastK)>=1 && cor_fastK<(-0.7)))
                          
                          
                        {
                          
                          
                          buy_or_sell_signal <- "HighDemand"
                        }else{
                          
                          #trend negative to positive
                          if((cor_val>(0) && sRSI<35 && sSD2==(-1) && sSD==1 && sfastK<0.2 && sfastD<0.1 && slast<s5MA && slast>spmedian && (sfastD2/sfastK2)>=1 && (sfastK/sfastD)>=1))
                            
                            
                          {
                            
                            
                            buy_or_sell_signal <- "PriceDownTrendChangeDownToUp"
                          }else{
                            
                            if(sfastK > sfastD && sRSI < 40 && sSD < 0 && sPctChangeClose < 0 && sPctChangeMA5 < 0 && sPctChangeMA20 < 0 && sPctChangeMA50< 0){
                              buy_or_sell_signal <- "TrendDown"
                            }else{
                              
                              if(sfastK<0.2 && sfastD<0.2 && cor_fastK<(-0.7) && cor_fastD<(-0.9) && sRSI < 50 && sSD < 0 && sPctChangeClose < 0 && sPctChangeMA5 < 0 && sPctChangeMA20 < 0 && sPctChangeMA50< 0 && slast<s5MA && slast<s20MA && slast<s50MA && volumechange>1 && cor_SMA20<(-0.9)){
                                buy_or_sell_signal <- "TrendDownHighVol"
                              }else{
                                
                                if(sRSI<45 && cor_fastD>0.7 && volumechange>2 && slast<s50MA && slast>s20MA && slast>s5MA && slast>spmedian){
                                  buy_or_sell_signal <- "PriceUpTrendUpHighVol"
                                  
                                }else{
                                  
                                  if(sRSI<75 && cor_fastD<(-0.7) && slast>s50MA && slast>s20MA && slast>s5MA && slast>spmedian && cor_SMA20>0.9){
                                    buy_or_sell_signal <- "PriceUpTrendUp"
                                    
                                  }else{
                                    
                                    if(sRSI<45 && s5MA>s20MA && s20MA>s50MA && slast<s5MA && sfastK>sfastD && sfastK<0.20 && smacd>smacdsignal){
                                      # Define trading signals
                                      buy_or_sell_signal <- "UptrendOversold"
                                    }else{
                                      
                                      if(sRSI>70 && s5MA<s20MA && s20MA<s50MA && slast>s5MA && sfastK<sfastD && sfastK>0.80 && smacd<smacdsignal){
                                        # Define trading signals
                                        buy_or_sell_signal <- "DowntrendOverbought"
                                      }else{
                                        
                                        #
                                        if(sRSI<45 && volumechange>1 && slast<s50MA && slast<s20MA && slast<s5MA && slast>spclose && sfastK<0.20 && sfastK>sfastD){
                                          buy_or_sell_signal <- "PriceDownTrendUpHighVol"
                                          
                                        }else{
                                          
                                          if(sRSI<45 && slast<s50MA && volumechange>1 && slast<s20MA && slast<s5MA && slast<spclose && (100*(spclose-slast)/spclose)>2 && sfastK<0.20 && sfastD<0.3){
                                            buy_or_sell_signal <- "PriceLowHighVol"
                                            
                                          }else{
                                            #  print("jere")
                                            
                                            if(sRSI<45 && slast<s50MA && slast<s20MA && slast<s5MA && slast<spclose && (100*(spclose-slast)/spclose)>2 && sfastK<0.20 && sfastD<0.3){
                                              buy_or_sell_signal <- "PriceLow"
                                              
                                            }else{
                                              
                                              if(slast<s50MA && slast<s20MA && slast<s5MA && slast<spclose){
                                                
                                                buy_or_sell_signal <- "PriceDownBelowMA"
                                                
                                              }else{
                                                
                                                if(slast<s50MA && slast<s20MA && s20MA<s50MA && smacd>0 && slast<spclose){
                                                  
                                                  buy_or_sell_signal <- "PriceDown20MAlessthan50MA"
                                                }else{
                                                  
                                                  if(slast<s50MA && slast<s20MA && slast<s5MA && s20MA<s50MA && sRSI<20 && (sfastD/sfastK)<=0.75  && get_pct_diff(spclose,slast)>1){
                                                    
                                                    buy_or_sell_signal <- "OversoldPriceUplessthanMA"
                                                  }else{
                                                    buy_or_sell_signal <- "Hold"
                                                  }
                                                }
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }   
                        
                      }
                    }
                  }
                }
              }    
            }
          }
        }
      }
      
    }
    
    #buy_or_sell_signal <- "Buy"
  } 
  else
  {
    #  && slast>s120MA 
    if((slast<s50MA && slast<s20MA && slast<spmedian && sSD>0 && cor_val>(0) && sRSI<35 && inflection_point==TRUE && slast<s5MA && s5MA<s20MA && stdi<0 && sdi<0 && cor_fastK<(-0.7) && volumechange>2))
      
      
    {
      
      #
      
      buy_or_sell_signal <- "PriceDownOverSellHighVol"
      
    }else{
      
      #low dip
      if((cor_val>(0) && sRSI<60 && sSD>0 && inflection_point==TRUE && slast>s120MA && slast<s5MA && sdi>0 && slast>s50MA && slast<spmedian && sfastD<0.20 && ((100*(s5MA-slast)/s5MA)>1)))
        
        
      {
        
        
        buy_or_sell_signal <- "PriceDownOverSold"
      }else{
        
        #steep dipslast<s120MA sdi<0 && 
        if((cor_val>(0) && sRSI<60 && inflection_point==TRUE && slast<spmedian  && slast<s5MA && cor_fastK>(0.7) && sfastD<0.40 && sfastD<0.40 && slast<s50MA && ((100*(s5MA-slast)/s5MA)>1) && ((100*(s50MA-slast)/s50MA)>3)))
          
          
        {
          
          
          buy_or_sell_signal <- "PriceDownTrendUp"
        }else{
          
          #fastK fastD cross-over&& ((100*(s5MA-slast)/s5MA)>1) && ((100*(s50MA-slast)/s50MA)>3)
          if((cor_val>(0) && sRSI<60 && sSD>0 && inflection_point==TRUE && slast<spmedian && cor_fastK<(-0.7) && slast<s5MA && (sfastK<0.1) && sfastD<0.20 && slast<s50MA && slast<s20MA))
            
            
          {
            
            
            buy_or_sell_signal <- "PriceDeclineTrendDown"
          }else{
            
            #downward short-term && ((100*(s5MA-slast)/s5MA)>1) && ((100*(s50MA-slast)/s50MA)>3)slast<s50MA && 
            if((cor_val>(0) && sRSI<60 && inflection_point==TRUE && slast>spmedian && slast>s5MA && cor_fastK>(0.9) && sfastK<0.20 && slast<s20MA && slast<s50MA && slast>s120MA && s50MA>s120MA && s50MA>s20MA && sSD>0))
            {
              
              buy_or_sell_signal <- "PriceShortDeclineTrendUp"
            }else{
              
              #trend changing upwards
              if((cor_val>(0) && sRSI<60 && slast>s120MA && sfastK<0.5 &&  inflection_point==TRUE && sTDI<0 && slast>spmedian && (100*abs(slast-s5MA)/s5MA)<5 && slast<s20MA && slast<s50MA && cor_fastK>(0.9) && (sfastK/sfastD)>=0.95))
              {
                buy_or_sell_signal <- "PriceDeclineTrendChangeUpwards"
              }
              else{
                #score>delta_thresh && 
                if (slast>spmedian && slast>s5MA && slast>s20MA && slast>s50MA && slast>s120MA && sRSI>50 &&  sfastK>0.9 && sfastD>0.8 && cor_fastK>(0.8) && (sfastK/sfastD)>1) {
                  buy_or_sell_signal <- "Sell"
                } else{
                  
                  if((cor_val>(0) && sRSI<60 && inflection_point==TRUE && slast>spmedian && (100*abs(slast-s5MA)/s5MA)<5 && cor_fastK<(-0.7) && sfastK<0.50 && slast<s20MA && slast<s50MA && stdi<0 && sSD<1 && (sfastD/sfastK)>1))
                  {
                    
                    buy_or_sell_signal <- "PriceDeclineTrendChangeUp"
                  }else{
                    # svolchg>3  && stdi>0 && sdi>0 
                    if((cor_val>(0) && sRSI<80 && sSD>0 && sfastK<0.5 && inflection_point==TRUE && slast>s120MA && slast>s5MA && slast>s50MA && slast>spmedian && (sfastD/sfastK)>=1 && cor_fastK<(-0.7)))
                      
                      
                    {
                      
                      
                      buy_or_sell_signal <- "PriceHighHighDemand"
                    }else{
                      
                      #trend negative to positive
                      if((cor_val>(0) && sRSI<35 && sSD2==(-1) && sSD==1 && sfastK<0.2 && sfastD<0.1 && slast<s5MA && slast>spmedian && (sfastD2/sfastK2)>=1 && (sfastK/sfastD)>=1))
                        
                        
                      {
                        
                        
                        buy_or_sell_signal <- "PriceDownTrendDownToUp"
                      }else{
                        
                        if(sfastK > sfastD && sRSI < 40 && sSD < 0 && sPctChangeClose < 0 && sPctChangeMA5 < 0 && sPctChangeMA20 < 0 && sPctChangeMA50< 0){
                          buy_or_sell_signal <- "TrendDown"
                        }else{
                          
                          if(sfastK<0.2 && sfastD<0.2 && cor_fastK<(-0.7) && cor_fastD<(-0.9) && sRSI < 50 && sSD < 0 && sPctChangeClose < 0 && sPctChangeMA5 < 0 && sPctChangeMA20 < 0 && sPctChangeMA50< 0 && slast<s5MA && slast<s20MA && slast<s50MA && volumechange>1 && cor_SMA20<(-0.9)){
                            buy_or_sell_signal <- "PriceDownTrendDownHighVol"
                          }else{
                            
                            if(sRSI<40 && cor_fastD>0.7 && volumechange>2 && slast<s50MA && slast>s20MA && slast>s5MA && slast>spmedian){
                              buy_or_sell_signal <- "PriceTrendUpHighVol"
                              
                            }else{
                              
                              if(sRSI<75 && cor_fastD<(-0.7) && slast>s50MA && slast>s20MA && slast>s5MA && slast>spmedian && cor_SMA20>0.9){
                                buy_or_sell_signal <- "PriceHigherTrendUp"
                                
                              }else{
                                
                                if(sRSI<45 && s5MA>s20MA && s20MA>s50MA && slast<s5MA && sfastK>sfastD && sfastK<0.20 && smacd>smacdsignal){
                                  # Define trading signals
                                  buy_or_sell_signal <- "Uptrend"
                                }else{
                                  
                                  if(sRSI>70 && s5MA<s20MA && s20MA<s50MA && slast>s5MA && sfastK<sfastD && sfastK>0.80 && smacd<smacdsignal){
                                    # Define trading signals
                                    buy_or_sell_signal <- "Downtrend"
                                  }else{
                                    
                                    if(sRSI<45 && volumechange>2 && slast<s50MA && slast<s20MA && slast<s5MA && slast>spclose && sfastK<0.20 && sfastK<0.20 && sfastK>sfastD){
                                      buy_or_sell_signal <- "PriceDownTrendUpHighVol"
                                      
                                    }else{
                                      
                                      if(slast<s50MA && slast<s20MA && slast<s5MA && slast<spclose){
                                        buy_or_sell_signal <- "PriceDownBelowMA"
                                      }else{
                                        if(slast<spclose){
                                          buy_or_sell_signal <- "PriceDown"
                                        }else{
                                          
                                          if(slast>spclose){
                                            buy_or_sell_signal <- "PriceUp"
                                          }else{
                                            
                                            if((cor_val<(0) && sRSI<60 && sSD>0 && inflection_point==TRUE && slast<spmedian && cor_fastK<(-0.7) && slast<s5MA && (sfastK<0.1) && sfastD<0.20 && slast<s50MA && slast<s20MA))
                                              
                                              
                                            {
                                              
                                              
                                              buy_or_sell_signal <- "Alert:PriceDeclineTrendDown"
                                            }
                                          }
                                        }
                                        
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }   
                    
                  }
                }
              }
            }
          }    
        }
      }
    }
  }
  #else{
  #   buy_or_sell_signal <- "Hold"
  
  #}
  # buy_or_sell_signal<-paste0(buy_or_sell_signal,":",round(cor_val,2),":",round(score,2),":",round(delta_thresh,2),":",slast)
  # }
  #     else{
  #    buy_or_sell_signal<- "Neutral"
  
  # }
  
  #dft<-data.frame(RSI=sRSI,StochK=sfastK,StochD=sfastD,MACD=smacd,MACDsignal=smacdsignal,s5MA,s20MA,s50MA,slast,spclose,WPR=swpr,CCI=scci)
  dft<-cbind(sRSI,sfastK,sfastD,smacd,smacdsignal,s5MA,s20MA,s50MA,s120MA,slast,spclose,swpr,scci,bollinger_bands$dn[length(bollinger_bands$dn)],bollinger_bands$up[length(bollinger_bands$up)])
  
  dft<-as.data.frame(dft)
  colnames(dft)<-c("RSI","StochK","StochD","MACD","MACDsignal","s5MA","s20MA","s50MA","s120MA","slast","spclose","WPR","CCI","BBDN","BBUP")
  save(dft,file="dft.Rda")
  if(sRSI<30 && sfastK>sfastD && sfastK<0.20 && smacd>smacdsignal && s5MA>s20MA){
    # Define trading signals
    buy_or_sell_signal <- paste0("BullOversoldUptrend:",buy_or_sell_signal)
  }else{
    
    if(sRSI>70 && s5MA<s20MA && s20MA<s50MA && sfastK<sfastD && sfastK>0.80 && smacd<smacdsignal){
      # Define trading signals
      buy_or_sell_signal <- paste0("BearOverboughtDowntrend:",buy_or_sell_signal)
    }else{
      if(is.na(sfastK2)==FALSE){
        if(sRSI<30 && sfastK2<0.20 && sfastK>0.2 && slast<s5MA && slast<s20MA){
          buy_or_sell_signal <- paste0("trendDownToUp: priceDown")
        }else{
          
          # if(sRSI<30 && sfastK2<0.20 && slast<s5MA && slast<s20MA && s50MA>s20MA && smacd>0){
          #buy_or_sell_signal <- paste0("trendDownToUp: priceDown")
          #}
          
        }
        
        
      }
      
      
    }
  }
  
  if(sRSI>50 && s5MA>s20MA && slast<s50MA && sRSI<70 && sfastK>=0.8 && slast<dft$BBUP && slast>s20MA && smacd>0 && scci>75){
    # Define trading signals
    buy_or_sell_signal <- paste0("UptrendLow:",buy_or_sell_signal)
  }else{
    
    if(sRSI<50 && s5MA<s20MA && slast<s20MA && sRSI>30 && sfastK>=0.2 && slast<dft$BBUP && slast>s20MA && smacd<0 && scci<(-25)){
      # Define trading signals
      buy_or_sell_signal <- paste0("DowntrendPriceLow:",buy_or_sell_signal)
    }else{
      
      if(sRSI>50 && s5MA<s20MA && slast>s50MA && sRSI<70 && sfastK>=0.8 && slast<dft$BBUP && slast<s20MA && slast<s5MA && smacd>0 && scci>10){
        # Define trading signals
        buy_or_sell_signal <- paste0("UptrendHigh:",buy_or_sell_signal)
      }
    }
  }
  
  if(FALSE){     
    dft$RSI<-ifelse(dft$RSI>70,1,ifelse(dft$RSI<30,-1,0))
    
    dft$StochKD<-ifelse(dft$StochK>dft$StochD,-1,ifelse(dft$StochK<dft$StochD,1,0))
    
    dft$StochK<-ifelse(dft$StochK>0.8,1,ifelse(dft$StochK<0.2,-1,0))
    dft$StochD<-ifelse(dft$StochD>0.75,1,ifelse(dft$StochD<0.25,-1,0))
    
    #The CCI usually falls in a channel of -100 to 100. A basic CCI trading system is: Buy (sell) if CCI rises above 100 (falls below -100) and sell (buy) when it falls below 100 (rises above -100).
    dft$CCI<-ifelse(dft$CCI<(-100),1,ifelse(dft$CCI>100,-1,0))
    
    #Readings from 0 to -20 are considered overbought. Readings from -80 to -100 are considered oversold. 
    #dft$WPR<-ifelse(dft$WPR>(-20) & dft$WPR<0,1,ifelse(dft$WPR>(-100) & dft$WPR<(-80),-1,0))
    dft$WPR<-ifelse(dft$WPR>0.8,(-1),ifelse(dft$WPR<0.2,1,0))
    
    dft$MACD<-ifelse(dft$MACD<dft$MACDsignal,1,ifelse(dft$MACD>dft$MACDsignal,-1,0))
    
    #dft$MIscore<-ifelse(dft$MACD>0 && dft$s50MA>dft$s200MA && dft$slast<dft$s5MA && dft$StochK<0.2 && dft$RSI<30 && dft$slast<df,-1,ifelse(dft$MACD<0 && #dft$slast>dft$s200MA && dft$slast>dft$s5MA,1,0))
    #dft$s20MA>dft$s50MA && dft$s5MA<dft$s20MA &&
    
    multindscore<-ifelse(dft$MACD>dft$MACDsignal &&  dft$slast<dft$s20MA &&  dft$slast<dft$s5MA && dft$StochK<=0.2 && dft$RSI<30 && dft$slast>dft$s50MA,-5,
                         ifelse(dft$MACD<0 && dft$slast>dft$s50MA && dft$slast>dft$s5MA,1,ifelse(dft$MACD>dft$MACDsignal &&  dft$slast>dft$s20MA && dft$StochK<0.2 && dft$RSI<30 && dft$spclose<dft$s20MA,-1,
                                                                                                 ifelse(dft$StochK<=0.2 && dft$RSI<35 && dft$slast<dft$s20MA,-1,ifelse(dft$StochK<=0.2 && dft$RSI<35 && dft$slast<dft$s20MA && dft$slast>dft$s5MA,-4,ifelse(dft$StochK<=0.2 && dft$RSI<35 && dft$slast<dft$s20MA,-1,0))))))
  }
  
  # print(head(dft))
  #buy_signal <- ifelse(current_price < ma_5 & current_price < ma_20 & rsi <= 40 & stochastic_k <= 0.2 & current_price <= bollinger_bands$dn, 1,
  #                    ifelse(current_price<ma_50 & ma_50<ma_200 & rsi >70 & stochastic_k >= 0.8 & current_price >= bollinger_bands$up, 1, 
  #                          ifelse(current_price> ma_50 & ma_50 > ma_200 & rsi <= 40 & stochastic_k <= 0.2 & current_price <= #bollinger_bands$dn, 1,0)))
  #sell_signal <- ifelse(current_price > ma_20 & rsi >35 & stochastic_k >= 0.8 & current_price >= bollinger_bands$up, 1,
  #                   ifelse(current_price>ma_50 & ma_50>ma_200 & rsi<=40 & stochastic_k <= 0.2 & current_price <= bollinger_bands$dn, 1, 
  #                         ifelse(current_price< ma_50 & ma_50<ma_200 & rsi>70 & stochastic_k>=0.8 & current_price >= bollinger_bands$up, 1,0)
  #                        ))
  #sell_signal <- ifelse(current_price > ma_20 & rsi >35 & stochastic_k >= 0.8 & current_price >= bollinger_bands$up, 1,
  #                 ifelse(current_price>ma_50 & ma_50>ma_200 & rsi<=40 & stochastic_k <= 0.2 & current_price <= bollinger_bands$dn, 0, 
  #                       ifelse(current_price> ma_5 & ma_50>ma_200 & rsi>70 & stochastic_k>=0.8 & current_price >= bollinger_bands$up, 1,0)
  #                ))
  
  multindscore<-ifelse(dft$slast < dft$s5MA & dft$slast<dft$s20MA & dft$RSI <= 40 & dft$StochK<=0.2 & dft$slast<=dft$BBDN, -1,
                       
                       
                       ifelse(dft$slast>dft$s20MA & dft$RSI > 40 & dft$StochK>=0.8 & dft$slast>=dft$BBUP,1,
                              
                              ifelse(dft$slast<dft$s50MA & dft$s50MA<dft$s120MA & dft$RSI > 70 & dft$StochK>=0.8 & dft$slast>=dft$BBUP,-1,
                                     ifelse(dft$slast>dft$s50MA & dft$s50MA>dft$s120MA & dft$RSI <=40 & dft$StochK<=0.2 & dft$slast<=dft$BBDN,-1,
                                            ifelse(dft$slast>dft$s5MA & dft$s50MA>dft$s120MA & dft$RSI > 70 & dft$StochK>=0.8 & dft$slast>=dft$BBUP,-1,0)))
                       ))
  
  #sell_signal <- ifelse(current_price > ma_20 & rsi >35 & stochastic_k >= 0.8 & current_price >= bollinger_bands$up, 1, 0)
  
  
  #multindscore<-sum(c(dft$RSI,dft$StochK,dft$StochKD,dft$StochD,dft$CCI,dft$WPR,dft$MACD,dft$MIscore),na.rm = TRUE)
  #multindscore<-sum(c(dft$RSI,dft$StochKD,dft$StochK,dft$MACD,dft$MIscore),na.rm = TRUE)
  freq_table<-table(c(dft$RSI,dft$StochKD,dft$StochK,dft$MACD,dft$MIscore))
  #multindscore<-names(freq_table)[which.max(freq_table)]
  #multindscore<-as.numeric(as.character(multindscore))
  multindscore<-multindscore[1]
  if(is.na(multindscore)==TRUE){
    
    multindscore=0
  }
  if(multindscore>3){
    # "Score(1):"
    buy_or_sell_signal <- paste0("Strong Sell: ",buy_or_sell_signal)
  }else{
    if(multindscore<(-3)){
      #Score(-1):
      buy_or_sell_signal <- paste0("Strong Buy: ",buy_or_sell_signal)
    }else{
      
      if(multindscore>0){
        # "Score(1):"
        buy_or_sell_signal <- paste0("Sell: ",buy_or_sell_signal)
      }else{
        if(multindscore<(0)){
          #Score(-1):
          buy_or_sell_signal <- paste0("Buy: ",buy_or_sell_signal)
        }else{
          #Score(0):
          buy_or_sell_signal <- paste0("Neutral: ",buy_or_sell_signal)
        }
        
      }
    }
  }
  x1=s20MA-(3*0.01*s20MA)
  x2=s50MA-(3*0.01*s50MA)
  x3=spclose-(3*0.01*spclose)
  #as.numeric(x2),
  buy_range=c(as.numeric(x1),as.numeric(x2),as.numeric(x3))
  buy_range<-round(buy_range,2)
  
  buy_range<-paste0(min(buy_range,na.rm=TRUE)[1]," : ", max(buy_range,na.rm=TRUE)[1]," (or less)")
  return(list(buy_or_sell_signal=buy_or_sell_signal,buy_range=buy_range))
  
  #return(buy_or_sell_signal)
}



get_stock_data <- function(symbol, start_date, end_date) {
  stock_data <- getSymbols(symbol, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
  stock_data <- data.frame(Date = index(stock_data), coredata(stock_data))
  stock_data <- stock_data %>%
    mutate(`20_MA` = SMA(Cl(stock_data), n = 20),
           `3_MA` = SMA(Cl(stock_data), n = 3),
           `50_MA` = SMA(Cl(stock_data), n = 50),
           `PClose`=lag(Close,1))
  return(stock_data)
}

predict_stock_price <- function(stock_data, premarket_price=NA, sentiment_score=NA) {
  # sentiment_score <- get_sentiment_score(news_headlines)
  
  
  model <- lm(stock_data$Adjusted ~ PClose + `3_MA`+`20_MA` + `50_MA`, data = stock_data)
  
  prediction <- predict(model, newdata = stock_data)
  return(prediction)
}

# Function to calculate Williams %R
williams_r <- function(high, low, close, n = 14) {
  highest_high <- runMax(high, n)
  lowest_low <- runMin(low, n)
  percent_r <- (highest_high - close) / (highest_high - lowest_low) * -100
  return(percent_r)
}

trading_strategy <- function(stock_data) {
  #getSymbols(symbol, src = "yahoo", from = "2024-01-01", to = "2025-02-27", auto.assign = FALSE) -> stock_data
  
  # print(head(stock_data))
  save(stock_data,file="stock_data.Rda")
  
  # Calculate indicators
  #  stock_data$WilliamsR <- williams_r(Cl(stock_data), Hi(stock_data), Lo(stock_data), n = 14)  # Williams %R
  stoch_values <-try(TTR::stoch(na.omit(stock_data[,c("High","Low","Close")])),silent=TRUE) #stoch(HLC(stock_data)), nFastK = 14, nFastD = 3, nSlowD = 3)  # Stochastic fastK & fastD
  if(is(stoch_values,"try-error")){
    #stoch_values<-1 #data.frame(nrow(stock_data),)
    stock_data$fastK <- 1 #stoch_values[, 1] 
    stock_data$fastD <- 1 #stoch_values[, 2]
    stock_data$RatioFastD_FastK <- 1 #stock_data$fastD / stock_data$fastK  # Ratio of fastD to fastK
    
  }else{
    
    stock_data$fastK <- stoch_values[, 1] 
    stock_data$fastD <- stoch_values[, 2]
    stock_data$RatioFastD_FastK <- stock_data$fastD / stock_data$fastK  # Ratio of fastD to fastK
    
  }
  wr_res <- try(williams_r(Cl(stock_data), Hi(stock_data), Lo(stock_data), n = 14),silent=TRUE)
  
  if(is(wr_res,"try-error")){
    
    stock_data$WilliamsR<-NA
  }else{
    
    stock_data$WilliamsR<-wr_res
  }
  
  rsi_res<- try(RSI(Cl(na.omit(stock_data)), n = 14),silent=TRUE)  # RSI
  if(is(rsi_res,"try-error")){
    stock_data$RSI <-NA
  }else{
    
    stock_data$RSI <-rsi_res
  }
  # Calculate daily price drop percentage
  stock_data$PriceChange <- (Cl(stock_data) - lag(Cl(stock_data), 1)) / lag(Cl(stock_data), 1) * 100
  
  stock_data<-na.omit(stock_data)
  # Generate buy and sell signals
  stock_data$BuySignal <- ifelse(
    stock_data$PriceChange < -3 &
      (lag(stock_data$WilliamsR, 1) > -80 & stock_data$WilliamsR <= -80) |
      (lag(stock_data$fastK, 1) < lag(stock_data$fastD, 1) & stock_data$fastK >= stock_data$fastD) |
      (lag(stock_data$RatioFastD_FastK, 1) < stock_data$RatioFastD_FastK) |
      (lag(stock_data$RSI, 1) < 30 & stock_data$RSI >= 30),
    1, 0)
  
  stock_data$SellSignal <- ifelse(
    stock_data$PriceChange > 3 &
      (lag(stock_data$WilliamsR, 1) < -20 & stock_data$WilliamsR >= -20) |
      (lag(stock_data$fastK, 1) > lag(stock_data$fastD, 1) & stock_data$fastK <= stock_data$fastD) |
      (lag(stock_data$RatioFastD_FastK, 1) > stock_data$RatioFastD_FastK) |
      (lag(stock_data$RSI, 1) > 70 & stock_data$RSI <= 70),
    1, 0)
  
  # Track buy and sell prices
  # stock_data$BuyPrice <- ifelse(stock_data$BuySignal == 1, Cl(stock_data), NA)
  # stock_data$SellPrice <- ifelse(stock_data$SellSignal == 1, Cl(stock_data), NA)
  
  # Forward-fill buy price until a sell occurs
  # stock_data$BuyPrice <- na.locf(stock_data$BuyPrice, na.rm = FALSE)
  
  # Compute returns when selling
  #  stock_data$TradeReturn <- ifelse(stock_data$SellSignal == 1 & !is.na(stock_data$BuyPrice),
  #                                 (stock_data$SellPrice - stock_data$BuyPrice) / stock_data$BuyPrice, NA)
  
  return(na.omit(stock_data))
}


get_stock_analysis<-function(watch_list,delta_thresh=0.5,period_val=3)
{
  
  Sys.sleep(0.5)
  watch_list<-unique(watch_list)
  
  stockres<-getQuote(watch_list,what=yahooQF(optnames))
  #load("C:/Users/karan/OneDrive/Documents/Productivity/mystockforecast/pso_params.Rda")
  
  stock_symbol="^GSPC"
  startrange=Sys.Date()-365
  endrange=Sys.Date()
  sp500_data<-quantmod::getSymbols(stock_symbol, from = "2023-01-01", to = Sys.Date(),warnings = FALSE,auto.assign = FALSE)
  
  
  buy_or_sell<-lapply(1:length(watch_list),function(i,delta_thresh,stockres){
    
    Sys.sleep(0.1)
    buy_or_sell_signal<-"Neutral"
    
    stock_symbol=watch_list[i]
    print(paste0("Analyzing ",stock_symbol))
    
    # print(stock_symbol)
    stock_data<-try(quantmod::getSymbols(stock_symbol[1], from = "2023-01-01", to = Sys.Date(),auto.assign = FALSE),silent=TRUE)
    #stock_data <- AAPL
    stock_data<-na.omit(stock_data)
    
    
    if(is(stock_data,"try-error")){
      print(stock_data)
      buy_or_sell_signal<-"Neutral"
      SMA20<-NA
      cor.with.sp500<-NA
      buy_range<-NA
      SMA5<-NA
      ytd_return<-NA
      
      stoch_res<-new("list")
      
      fastK<-NA
      fastD<-NA
      SMA5_volume=NA
      daily_mean <- NA
      weekly_mean <- NA
      monthly_mean <- NA
      
      daily_sd <- NA
      weekly_sd <- NA
      monthly_sd <- NA
      sharpe_ratio <- NA
      
      var_95 <-NA
      pivot_point<-NA
      forecast_trend<-NA
      forecast_range<-NA
      cor_val<-NA
      sTDI<-NA
      sRSI<-NA
      sfastD<-NA
      stdi<-NA
      sdi<-NA
      sfastK<-NA
      smacd<-NA
      smacdsignal<-NA
      smom<-NA
      swpr<-NA
      trading_res<-NA
      buy_signal<-NA #trading_res$BuySignal
      sell_signal<-NA
    }
    else
    {
      if(nrow(stock_data)>30){
        
        
        
        
        cnames1<-colnames(stock_data)
        
        cnames1<-gsub(cnames1,pattern=paste0(stock_symbol,"\\."),replacement="")
        
        colnames(stock_data)<-cnames1
        
        colnames(sp500_data)<-cnames1
        
        stock_cur<-cbind(stockres$Open[i],stockres$High[i],stockres$Low[i],stockres$Last[i],stockres$Volume[i],stockres$Last[i])
        stock_cur<-as.data.frame(stock_cur)
        colnames(stock_cur)<-cnames1
        
        rownames(stock_cur)<-Sys.Date()
        
        
        stock_data<-as.data.frame(stock_data)
        sp500_data<-as.data.frame(sp500_data)
        
        check_miss<-which(is.na(stock_data$Close)==TRUE)
        
        if(length(check_miss)>0){
          
          stock_data$Close[check_miss]<-mean(stock_data$Close,na.rm=TRUE)  
          
        }
        stock_data_orig<-stock_data
        
        stock_data<-as.data.frame(stock_data)
        sp500_data<-as.data.frame(sp500_data)
        
        df1<-cbind(rownames(stock_data),stock_data[,ncol(stock_data)])
        #forecast price
        forecast_res<-forecast_stock_price(df1)
        
        forecast_trend<-forecast_res$forecast_trend
        
        forecast_range<-forecast_res$forecast_range
        
        
        
        # print(tail(stock_data))
        # print(tail(sp500_data))
        #calculate Pearson correlation
        cor.with.sp500<-try(round(cor(stock_data$Close,sp500_data$Close[which(rownames(sp500_data)%in%rownames(stock_data))],use = "pairwise.complete.obs"),3),silent=TRUE)
        
        if(is(cor.with.sp500,"try-error")){
          
          cor.with.sp500<-NA
        }
        
        stock_data<-stock_data%>%as.data.frame%>%rbind(stock_cur)
        
        
        #  stock_data<-rbind(stock_data,stock_cur)
        
        stock_data<-as.xts(stock_data)
        
        #stock_data<-unique(stock_data)
        rnames1<-rownames(stock_data)
        rnames1<-gsub(rnames1,pattern="^X",replacement="")
        rnames1<-gsub(rnames1,pattern="\\.",replacement="-")
        
        
        rownames(stock_data)<-rnames1
        
        trading_res<-try(trading_strategy(stock_data),silent=TRUE)
        if(is(trading_res,"try-error")){
          buy_signal<-0
          sell_signal<-0
        }else{
        trading_res<-trading_res[nrow(trading_res),c("BuySignal","SellSignal")]
        buy_signal<-trading_res$BuySignal
        sell_signal<-trading_res$SellSignal
        }
        #  stock_data<-stock_data_orig
        # print("here1")
        # Calculate daily returns
        daily_returns <- 1*dailyReturn(Cl(stock_data[which(time(stock_data)>"2024-01-01"),]))
        
        weekly_returns<-1*weeklyReturn(Cl(stock_data[which(time(stock_data)>"2024-01-01"),]))
        
        monthly_returns<-1*monthlyReturn(Cl(stock_data[which(time(stock_data)>"2024-01-01"),]))
        
        # print("here2")
        
        # Calculate YTD return
        ytd_return <- round(Return.cumulative(daily_returns)*100,2)
        
        # Calculate mean returns
        daily_mean <- round(mean(daily_returns*100,na.rm=T),2)
        weekly_mean <- round(mean(weekly_returns*100,na.rm=T),2)
        monthly_mean <- round(mean(monthly_returns*100,na.rm=T),2)
        
        # Calculate standard deviation (volatility)
        daily_sd <- round(sd(daily_returns,na.rm=T),2)
        weekly_sd <- round(sd(weekly_returns,na.rm=T),2)
        monthly_sd <- round(sd(monthly_returns,na.rm=T),2)
        
        #df$trend<-ifelse(mean(TDI(Ad(stock_data),period_val)$di,na.rm=TRUE)>(0),1,-1)
        
        
        # Calculate Sharpe ratio (assuming risk-free rate is 0.05; 5% Source: https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=yield)
        risk_free_rate <- 0.01
        sharpe_ratio <- round(SharpeRatio.annualized(daily_returns, Rf = risk_free_rate),2)
        
        # Calculate Value at Risk (VaR)
        var_95 <- round(VaR(daily_returns, p = 0.95, method = "historical"),2)*100
        
        pivot_point<-get_resistance_support(stock_data=stock_data,tperiod=60)
        
        smom<-momentum(Cl(stock_data), n = 3)
        smom<-smom[nrow(stock_data)]
        
        df<-stock_data
        #  cor_val<-cor(seq(1,nrow(df)),df$Close)
        
        cor_val<-cor(seq(1,5),df$Close[((length(df$Close)-4):length(df$Close))])
        #print(cor_val)
        if(is.na(cor_val)==TRUE){
          
          cor_val=-1
        }
        SecondDerivative <- sign(diff(diff(Cl(stock_data))))
        
        
        #print(head(SecondDerivative))
        
        #print(length(SecondDerivative))
        #print(head(df))
        #NA,NA,
        
        df$SecondDerivative <-c(SecondDerivative)
        #print(head(df))
        #print(tail(stock_data))
        
        rsi_res <- try(RSI((Cl(stock_data)), n = 14),silent=TRUE) #)max(c(nrow(stock_data),20))
        
        if(is(rsi_res,"try-error")){
          
          df$RSI <-NA
        }else{
          df$RSI <-rsi_res
        }
        
        df$AvgVolume <- rollapply(Vo(stock_data), width = 20, FUN = mean, align = "right", fill = NA)
        df$VolumeChange <- Vo(stock_data) / df$AvgVolume
        
        #stochastics
        #stoch_res<-TTR::stoch(stock_data[,c("High","Low","Close")]) #,stockres[i,c("High","Low","Last")]))
        stoch_res<-new("list")
        #stochastics(nrow(stock_data)-30):nrow(stock_data)
        stoch_res<-try(TTR::stoch(na.omit(stock_data[,c("High","Low","Close")])),silent=TRUE)
        #(nrow(stock_data)-30):nrow(stock_data)
        macd_res<-try(TTR::MACD(na.omit(stock_data[,c("Close")])),silent=TRUE)
        
        CCI_res <- try(TTR::CCI(na.omit(stock_data[,c("High","Low","Close")])),silent=TRUE)
        #WPR_res <- try(TTR::WPR(na.omit(stock_data[,c("High","Low","Close")])),silent=TRUE) 
        WPR_res<- try(williams_r(Cl(stock_data), Hi(stock_data), Lo(stock_data), n = 14),silent=TRUE)
        
        bollinger_bands <- try(BBands(Cl(stock_data), n = 20),silent=TRUE)
        
        if(is(stoch_res,"try-error")){
          
          fastK<-NA
          fastD<-NA
          fastK2<-NA
          fastD2<-NA
          cor_fastK<-NA
          cor_fastD<-NA
          
        }else{
          fastK<-round(stoch_res$fastK[length(stoch_res$fastK)],2)
          fastD<-round(stoch_res$fastD[length(stoch_res$fastD)],2)
          fastK2<-round(stoch_res$fastK[length(stoch_res$fastK)-1],2)
          fastD2<-round(stoch_res$fastD[length(stoch_res$fastD)-1],2)
          
          fastk_vec<-stoch_res$fastK[(length(stoch_res$fastK)-2):length(stoch_res$fastK)]
          fastd_vec<-stoch_res$fastD[(length(stoch_res$fastD)-2):length(stoch_res$fastD)]
          cor_fastK<-cor(c(1:3),fastk_vec,use="p")
          cor_fastD<-cor(c(1:3),fastd_vec,use="p")
        }
        
        if(is(macd_res,"try-error")){
          smacd<-NA
          smacdsignal<-NA
        }else{
          
          smacd<-macd_res$macd[nrow(macd_res)]
          smacdsignal<-macd_res$signal[nrow(macd_res)]
        }
        
        if(is(CCI_res,"try-error")){
          scci<-NA
          
        }else{
          
          scci<-CCI_res[nrow(CCI_res)]
          
        }
        
        if(is(WPR_res,"try-error")){
          swpr<-NA
          
        }else{
          
          swpr<-WPR_res[nrow(WPR_res)]
          
        }
        
        if(is(bollinger_bands,"try-error")){
          
          bollinger_bands$dn<-NA
          bollinger_bands$up<-NA
        }
        
        sfastD=fastD
        sfastK=fastK
        sfastD2=fastD2
        sfastK2=fastK2
        
        SMA5_volume=try(rollapply(Vo(stock_data), width = 20, FUN = mean, align = "right"),silent=TRUE) #try(SMA(stock_data$Volume,n=5),silent=TRUE)
        if(is(SMA5_volume,"try-error")){
          
          SMA5_volume=NA
        }else{
          
          SMA5_volume<-SMA5_volume[nrow(SMA5_volume)]
        }
        AvgVolume=SMA5_volume #df$AvgVolume[length(df$AvgVolume)] # 
        volumechange <- stockres$`Volume`[i]/AvgVolume #Vo(stock_data) / AvgVolume
        # Calculate moving averages
        SMA20_res <- try(SMA(Cl(stock_data), n = 20),silent=TRUE)
        
        
        
        sPctChangeClose <- Delt(Cl(stock_data))
        sPctChangeMA5 <- try(Delt(SMA(Cl(stock_data), n = 5)),silent=TRUE)
        
        if(is(sPctChangeMA5,"try-error")){
          sPctChangeMA5<-NA
        }
        
        if(is(SMA20_res,"try-error")){
          SMA20<-NA
          sPctChangeMA20 <-sPctChangeMA5
        }else{
          
          stock_data$SMA20<-SMA20_res
          SMA20<-round(stock_data$SMA20[nrow(stock_data)],3)
          sPctChangeMA20 <- Delt(SMA(Cl(stock_data), n = 20))
        }
        
        cor_SMA20<-try(cor(c(1:5),SMA20_res[c((length(SMA20_res)-4):length(SMA20_res))],use="p"),silent = TRUE)
        
        if(is(cor_SMA20,"try-error")){
          
          cor_SMA20<-NA
        }
        
        SMA50_res <- try(SMA(Cl(stock_data), n = 50),silent=TRUE)
        
        if(is(SMA50_res,"try-error")){
          SMA50<-NA
          sPctChangeMA50<-sPctChangeMA20
        }else{
          
          stock_data$SMA50<-SMA50_res
          # stockres$`50-day MA`[i]<-stock_data$SMA50[nrow(stock_data$SMA50)]
          SMA50<-round(stock_data$SMA50[nrow(stock_data)],1)
          sPctChangeMA50 <- try(Delt(SMA(Cl(stock_data), n = 50)),silent=TRUE)
          
          if(is(sPctChangeMA50,"try-error")){
            sPctChangeMA50<-sPctChangeMA20
          }
          
        }
        
        SMA120_res <- try(SMA(Cl(stock_data), n = 120),silent=TRUE)
        
        if(is(SMA120_res,"try-error")){
          SMA120<-SMA50
        }else{
          
          stock_data$SMA120<-SMA120_res
          # stockres$`50-day MA`[i]<-stock_data$SMA50[nrow(stock_data$SMA50)]
          SMA120<-round(stock_data$SMA120[nrow(stock_data)],1)
        }
        
        
        SMA5_res <- try(SMA(Cl(stock_data), n = 5),silent=TRUE)
        
        if(is(SMA5_res,"try-error")){
          SMA5<-SMA20
        }else{
          
          stock_data$SMA5<-SMA5_res
          # stockres$`50-day MA`[i]<-stock_data$SMA50[nrow(stock_data$SMA50)]
          SMA5<-round(stock_data$SMA5[nrow(stock_data)],1)
        }
        
        #  print(head(stock_data))
        
        tdi_res<-TTR::TDI((stock_data$Adjusted),n =period_val)%>%as.data.frame()
        #    print(head(tdi_res))
        #    print(tail(tdi_res))
        
        #forecast_trend<-ifelse(tdi_res$di>(0),1,-1)
        if(FALSE){               
          previous_3daytrend=mean(forecast_trend[(length(forecast_trend)-3):(length(forecast_trend)-1)],na.rm=TRUE)
          
          if(previous_3daytrend<0 && forecast_trend[length(forecast_trend)]>0){
            
            inflection_point=TRUE
          }else{
            inflection_point=FALSE
          }
        }      
        #  forecast_trend<-forecast_trend[length(forecast_trend)]
        
        #if(nrow(stock_data)<200){
        
        #stock_data$SMA5 <- SMA(Cl(stock_data), n = min(200,nrow(stock_data)))
        #stockres$`5-day MA`[i]<-stock_data$SMA5[nrow(stock_data$SMA5)]
        
        
        
        #     print(nrow(df))
        #     print(nrow(stock_data))
        #    print(nrow(tdi_res))
        
        # Calculate second derivative to identify inflection points
        
        
        
        TDI_res<-tdi_res #TDI((stock_data$Adjusted),7)%>%as.data.frame() #TDI(df$DailyReturn,3) #TDI(Cl(stock_data),7) #TDI(df$DailyReturn,7) & df$VolumeChange>1
        #TDI_res$di>0 & 
        df$TDI <- ifelse(TDI_res$tdi>(0) & TDI_res$di>(-50) & df$SecondDerivative>0 & df$RSI<50,1,ifelse(TDI_res$tdi>(100) & TDI_res$di>(100) & df$SecondDerivative>0 & df$RSI>50,100,-1))
        df$tdi<-TDI_res$tdi
        df$di<-TDI_res$di
        sTDI=df$TDI[nrow(df)]
        stdi=df$tdi[nrow(df)]
        sdi=df$di[nrow(df)]
        sSD=df$SecondDerivative[nrow(df)]                
        # buy_or_sell_signal<-try(get_buy_sell_signal(stockres$`Last`[i],stockres$`P. Close`[i],SMA20,SMA50,SMA5,delta_thresh,stockres$Ask[i],stockres$Bid[i],fastK,fastD,sharpe_ratio,monthly_mean,forecast_trend),silent=FALSE)
        if(FALSE){
          buy_or_sell_signal<-try(get_buy_sell_signal(slast=stockres$`Last`[i],spclose=stock_data$High[nrow(stock_data)],s5MA=SMA5,s20MA=SMA20,s50MA=SMA50,s120MA=SMA120,
                                                      delta_thresh=delta_thresh,forecast_trend,previous_postion="Neutral",volume_change,inflection_point),silent=FALSE)
        }
        
        #print(volumechange)
        sRSI=df$RSI[nrow(df)]
        sTDI=df$TDI[nrow(df)]
        
        buy_or_sell_signal<-try(get_buy_sell_signal(slast=stockres$`Last`[i], spclose=stockres$`P. Close`[i],spmedian=df$High[nrow(df)], sphigh=stock_data$High[nrow(stock_data)],s5MA=SMA5,s20MA=SMA20,s50MA=SMA50,s120MA=SMA120,sRSI=df$RSI[nrow(df)],sTDI=df$TDI[nrow(df)],cor_val=cor_val,sSD=df$SecondDerivative[nrow(df)],sSD2=df$SecondDerivative[nrow(df)-1],sfastD=sfastD,sfastD2=sfastD2,sfastK=sfastK,sfastK2=sfastK2,cor_fastK=cor_fastK,cor_fastD=cor_fastD,stdi=df$tdi[nrow(df)],sdi=df$di[nrow(df)],sPctChangeClose[nrow(df)],sPctChangeMA5[nrow(df)],sPctChangeMA20[nrow(df)],sPctChangeMA50[nrow(df)],weights=NA,volumechange,cor_SMA20,smacd,smacdsignal,scci,swpr,bollinger_bands),silent=FALSE)
        
        # signal_res<-get_buy_sell_signal(slast=last_price,spclose=prev_close,s5MA=ma5,s20MA=ma20,s50MA=ma50,s120MA=ma120,
        #    delta_thresh=delta_thresh,forecast_trend,previous_postion,volume_change,inflection_point)
        
        df$TDI <- ifelse(TDI_res$di>(0) & df$SecondDerivative>0,1,-1)          
        if(is(buy_or_sell_signal,"try-error")){
          buy_or_sell_signal<-NA
          buy_range<-NA
        }else{
          
          buy_range<-buy_or_sell_signal$buy_range
          buy_or_sell_signal<-buy_or_sell_signal$buy_or_sell_signal
          
        }
        
        Sys.sleep(0.2)
        
        #print(cor_val)
        
        # print(buy_or_sell_signal)
        #buy_or_sell_signal<-paste0(buy_or_sell_signal,"_",cor.with.sp500,":",SMA20,"-",buy_range)
        
        
      }
      else{
        
        buy_or_sell_signal<-"Neutral"
        SMA20<-NA
        cor.with.sp500<-NA
        buy_range<-NA
        SMA5<-NA
        ytd_return<-NA
        
        stoch_res<-new("list")
        
        fastK<-NA
        fastD<-NA
        SMA5_volume=NA
        daily_mean <- NA
        weekly_mean <- NA
        monthly_mean <- NA
        
        daily_sd <- NA
        weekly_sd <- NA
        monthly_sd <- NA
        sharpe_ratio <- NA
        
        var_95 <-NA
        forecast_trend<-NA
        forecast_range<-NA
        cor_val<-NA
        sTDI<-NA
        sRSI<-NA
        smacd<-NA
        smacdsignal<-NA
        pivot_point<-NA
        smom<-NA
        swpr<-NA
        buy_signal<-NA #trading_res$BuySignal
        sell_signal<-NA
      }
    }
    
    return(list(buy_or_sell_signal=buy_or_sell_signal,cor.with.sp500,SMA20,SMA5,buy_range,fastK,fastD,SMA5_volume,ytd_return,daily_mean,
                weekly_mean,monthly_mean,daily_sd,weekly_sd,monthly_sd,sharpe_ratio,var_95,forecast_range,sTDI,round(sRSI,2),
                smacd,smacdsignal,pivot_point,smom,swpr,buy_signal,sell_signal))
    
  },delta_thresh=delta_thresh,stockres=stockres)
  
  
  
  stockres$`% Change From 52-week Low`<-round(stockres$`% Change From 52-week Low`*100,1)
  stockres$`% Change From 52-week High`<-round(stockres$`% Change From 52-week High`*100,1)
  stockres$`% Change`<-round(stockres$`% Change`,1)
  stockres$`% Change From 50-day MA`<-round(stockres$`% Change From 50-day MA`*100,1)
  stockres$`% Change From 200-day MA`<-round(stockres$`% Change From 200-day MA`*100,1)
  
  buy_or_sell_res<-buy_or_sell
  
  # print(head(buy_or_sell))
  
  #save(buy_or_sell,stockres,watch_list,file="C:/Users/karan/OneDrive/Documents/Productivity/mystockforecast/buy_or_sell.Rda")
  
  buy_or_sell<-t(sapply(buy_or_sell,as.data.frame))
  buy_or_sell<-as.data.frame(buy_or_sell)
  colnames(buy_or_sell)<-c("buy_or_sell","cor.with.sp500","SMA20","SMA5","buy_range","fastK","fastD","SMA5_volume",
                           "ytd_return","daily_mean","weekly_mean","monthly_mean","daily_sd","weekly_sd","monthly_sd","sharpe_ratio","var_95","forecast_range","forecast_trend","RSI","smacd","smacdsignal","pivot_point","momentum","WilliamR","buy_signal","sell_signal")
  
 
 # stockres<-cbind(rownames(stockres),stockres)
  #cnames1<-colnames(stockres)
  #cnames1[1]<-"Symbol"
 # colnames(stockres)<-cnames1
  
  #print(head(stockres))
  #print(dim(stockres))
  
  # print(head(buy_or_sell))
  # print(dim(buy_or_sell))
  buy_or_sell<-cbind(watch_list,buy_or_sell)
  
  
 # colnames(stockres_final)<-c(cnames1,colnames(buy_or_sell))
  
  stockres_final<-merge(stockres,buy_or_sell,by.x="Symbol",by.y="watch_list")%>%unique() ##cbind(stockres,buy_or_sell) #$SMA20,buy_or_sell$buy_or_sell,buy_or_sell$cor.with.sp500,buy_or_sell$buy_range)
  
  
  stockres_final$buy_or_sell<-as.factor(unlist(stockres_final$buy_or_sell))
  
  forecast.range<-NA #sapply(forecast_res,unlist) #[1,]
  #cor.with.sp500<-as.numeric(as.character(sapply(forecast_res,unlist)[2,]))
  stockres_final<-cbind(stockres_final,forecast.range) #,cor.with.sp500)
  #return(stockres)
  #stockres_final<-cbind(stockres_final,trading_res)
  #})
  #,"forecast.range","cor.with.sp500"
  
  
  
  stockres_final1<-stockres_final[,c("Symbol","Name","Last","P. Close","% Change","Low","High","52-week Low","52-week High","SMA20","50-day MA","SMA5","% Change From 50-day MA","% Change From 200-day MA","% Change From 52-week Low","% Change From 52-week High","Price/EPS Estimate Current Year","EPS Forward","EPS Current Year","buy_or_sell","cor.with.sp500","buy_range","fastK","fastD","RSI","SMA5_volume","Ave. Daily Volume","Volume","Ask","Bid","Ave. Analyst Rating","YTD Return","ytd_return","daily_mean","weekly_mean","monthly_mean","daily_sd","weekly_sd","monthly_sd","sharpe_ratio","var_95","pivot_point","forecast_range","forecast_trend","Market Capitalization", "Earnings Timestamp","momentum","WilliamR","buy_signal","sell_signal")]%>%mutate(MarketCapCat=ifelse(`Market Capitalization`>10*10^9,"Large",ifelse(`Market Capitalization`>2*10^9 & `Market Capitalization`<10*10^9,"Mid",                                                                      ifelse(`Market Capitalization`>300*10^6 & `Market Capitalization`<2*10^9,"Small",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ifelse(`Market Capitalization`>50*10^6 & `Market Capitalization`<300*10^6,"Micro","Nano")))))
  
  
  
  #,"forecast.range","cor.with.sp500"
  
  #stockres_final1<-cbind(watch_list,stockres_final1)
  
  colnames(stockres_final1)<-c("Symbol","Name","Last","P.Close","%Change","Low","High","52 week Low","52 week High","20 day MA","50 day MA","5 day MA","% Change.50-day.MA","% Change.200-day.MA","% Change.52-week.Low","% Change.52-week.High","Price.EPS","EPS Forward","EPS Current Year","Trade.Signal","Cor.withSP500","Buy.Range","fastK","fastD","RSI","SMA5_volume","Ave. Daily Volume","Volume","Ask","Bid","Ave. Analyst Rating","YTD Return","ytd_return","daily_mean","weekly_mean","monthly_mean","daily_sd","weekly_sd","monthly_sd","sharpe_ratio","var_95","pivot_point","Forecast Range","Forecast Trend","Market Capitalization","Earnings Timestamp","momentum","WilliamR","buy_signal","sell_signal","MarketCapCat") 
  
  stockres_final1<-stockres_final1%>%rowwise()%>%mutate(fastDKratio=ifelse(is.na(fastD)==FALSE,round(as.numeric(as.character(fastD))/as.numeric(as.character(fastK)),2),NA)) #%>%filter(Price.EPS>0 & Cor.withSP500>0.5 & momentum<3 & fastDKratio>1.25)
  
  stockres_final1<-stockres_final1%>%dplyr::rename(R.SP=Cor.withSP500,Rating=Trade.Signal,`Buy Range`=Buy.Range)%>%mutate(Overbought.vs.Oversold=ifelse(fastK<=0.20 & RSI<40,"Oversold",ifelse(fastK>=0.8 & RSI>70,"Overbought","Neutral")),Price.EPS=round(Price.EPS,2),fastD=as.numeric(as.character(fastD)),Popularity.Level=ifelse(Volume>1.5*`Ave. Daily Volume` & Volume>SMA5_volume,"High",ifelse(Volume>1.5*`Ave. Daily Volume` | Volume>SMA5_volume,"Medium","Low")))%>%select(Symbol,MarketCapCat,Overbought.vs.Oversold,`Forecast Range`,`Forecast Trend`,Rating,R.SP,`Buy Range`,Popularity.Level,ytd_return,Last,P.Close,Low,High,Ask,Bid,`EPS Forward`,`EPS Current Year`,`%Change`,`20 day MA`,`50 day MA`,`5 day MA`,fastK,fastD,`Ave. Analyst Rating`,SMA5_volume,`Ave. Daily Volume`,Volume,`52 week Low`,`52 week High`,Name,daily_mean,weekly_mean,monthly_mean,daily_sd,weekly_sd,monthly_sd,sharpe_ratio,var_95,pivot_point,RSI,fastDKratio,R.SP,Price.EPS,WilliamR,buy_signal,sell_signal)
  
  #,Forecast,Cor.withSP500 ,Forecast,Cor.withSP500,Overbought.vs.Oversold=as.factor(Overbought.vs.Oversold)
  #,`Overbought vs Oversold`
  #mat3<-stockres_final3b%>%mutate(`Year-to-Date Return`=as.numeric(as.character(ytd_return)),`Overbought vs Oversold`=as.factor(Overbought.vs.Oversold))%>%select(Symbol,MarketCapCat,Rating,Popularity.Level,Last,P.Close,`%Change`,`5 day MA`,`20 day MA`,`50 day MA`,`Buy Range`,Volume,`Year-to-Date Return`,var_95,pivot_point,`Overbought vs Oversold`)%>%mutate(MarketCapCat=as.factor(MarketCapCat),Rating=as.factor(Rating),Popularity.Level=as.factor(Popularity.Level),var_95=as.numeric(as.character(var_95)))%>%rename(`Popularity Level`=Popularity.Level,MarketCap=MarketCapCat,`Value.At.Risk(%)`=var_95,`50-Day Pivot Point`=pivot_point)
  
  stockres_final1<-stockres_final1%>%mutate(`Year-to-Date Return`=as.numeric(as.character(ytd_return)),
                                            `Overbought vs Oversold`=as.factor(Overbought.vs.Oversold))%>%select(Symbol,MarketCapCat,Rating,Popularity.Level,
                                                                                                                 Last,P.Close,Low,High,`%Change`,`5 day MA`,`20 day MA`,`50 day MA`,
                                                                                                                 `Buy Range`,Volume,`Year-to-Date Return`,var_95,pivot_point,`Overbought vs Oversold`,
                                                                                                                 `Forecast Range`,RSI,fastDKratio,R.SP,Price.EPS,WilliamR,buy_signal,sell_signal)%>%mutate(MarketCapCat=as.factor(MarketCapCat),Rating=as.factor(Rating),Popularity.Level=as.factor(Popularity.Level),var_95=as.numeric(as.character(var_95)))%>%dplyr::rename(`Popularity Level`=Popularity.Level,MarketCap=MarketCapCat,`Risk(%)`=var_95,`Pivot.Point`=pivot_point,`Price.Status`=`Overbought vs Oversold`)%>%mutate(RSI.Check=ifelse(RSI>50 & RSI<70,1,0),Pivot.Check=ifelse(Last>Pivot.Point,1,-1),Pivot.Low.Check=ifelse(Low>Pivot.Point,1,-1),MA5.Check=ifelse(Last>`5 day MA`,1,-1),MA20.Check=ifelse(Last>`20 day MA`,1,-1),MA50.Check=ifelse(Last>`50 day MA`,1,-1),Prev.Check=ifelse(100*((High-`P.Close`)/`P.Close`)<(-1),-1,1))
  
  
  return(stockres_final1)
}


get_stock_entry_point<-function(xlow,xhigh,xlast){
  
  val<-max(c((xlast)*0.97,xlow))
}

get_pct_diff<-function(val1,val2){return(round(100*(val2-val1)/val1,2))}

get_premarket_health<-function(symbol){
  
  premarket_data<-get_stockdata_alpaca(symbol,"1minute")%>%rowwise()%>%mutate(pct_diff=get_pct_diff(low,high))
  
  p1=premarket_data%>%filter(timestamp>paste0(Sys.Date()," 9:00:00") & timestamp<paste0(Sys.Date()," 9:30:00"))
  
  c1<-try(cor(seq(1,nrow(p1)),p1$vwap),silent=TRUE)
  
  if(is(c1,"try-error")){
    c1=-1
  }
  
  premarket_status<-ifelse(c1>0.5,1,0)
}



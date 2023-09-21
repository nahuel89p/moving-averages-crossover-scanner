import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output, State
import numpy as np
from finta import TA
import yfinance as yf
import copy
from plotly.subplots import make_subplots
import dash_table
import random
from dash.exceptions import PreventUpdate

def moving_averages(btc,tma1,tma2,ma1,ma2): 
    m1=50
    global matriz
    if tma1 == "sma":
        btc['ma1'] = btc['close'].rolling(window=ma1).mean()
    elif tma1 == "ema":
        btc['ma1'] = btc['close'].ewm(span=ma1, adjust=False).mean()
    elif tma1 == "wma":
        weights = np.arange(1,ma1+1) #this creates an array with integers 1 to 10 included
        btc['ma1'] = btc['close'].rolling(ma1).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    elif tma1 == "dema":
        ema1 = btc['close'].ewm(span=ma1, adjust=False).mean()
        ema2 = ema1.ewm(span=ma1, adjust=False).mean()
        btc['ma1'] = 2 * ema1 - ema2
    elif tma1 == "tema":
        btc['ma1'] = TA.TEMA(btc, ma1)
    elif tma1 == "trima":
        btc['ma1'] = TA.TRIMA(btc, ma1)     
    elif tma1 == "vama":
        btc['ma1'] = TA.VAMA(btc, ma1)              
    elif tma1 == "kama":
        btc['ma1'] = TA.KAMA(btc, er= 30, ema_fast = int(round(min(ma1,ma2))/10) , ema_slow =int(round(max(ma1,ma2))/10)  , period=40)   
    elif tma1 == "zlema":
        btc['ma1'] = TA.ZLEMA(btc, ma1)
    elif tma1 == "evwma":
        btc['ma1'] = TA.EVWMA(btc, ma1) 
    elif tma1 == "clwma":
        btc['weights'] = btc['high'] -btc['low'] 
        btc['ma1'] = get_weighted_average(btc,ma1,"close","weights").values
    elif tma1 == "qclwma":
        btc['weights'] = pow((btc['high'] -btc['low']) ,4).axis =1
        btc['ma1'] = get_weighted_average(btc,ma1,"close","weights").values    
    elif tma1 == "vwma":
        btc['ma1'] = get_weighted_average(btc,ma1,"close","volume").values
    elif tma1 == "3cvwma":
        btc['ma1'] = vqvwma(btc.close,btc.volume,ma1*m1,3)
    elif tma1 == "2cvwma":
        btc['ma1'] = vqvwma(btc.close,btc.volume,ma1*m1,2)
    elif tma1 == "minmax_c":
        min1=btc['close'].rolling(ma1).min()
        max1=btc['close'].rolling(ma1).max()
        btc['ma1'] = (min1 +max1) / 2
    elif tma1 == "minmax_o":
        min1=btc['open'].rolling(ma1).min()
        max1=btc['open'].rolling(ma1).max()
        btc['ma1'] = (min1 +max1) / 2

    elif tma1 == "minmaxlh":
        min1=btc['low'].rolling(ma1).min()
        max1=btc['high'].rolling(ma1).max()
        btc['ma1'] = (min1 +max1) / 2

    elif tma1 == "osciladorvol":
        btc['ma1'] = btc['oscilador'].ewm(span=ma1, adjust=False).mean()
        
    elif tma1 == "dynamicminmax1":    
        btc['ma1'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma1,1)
    elif tma1 == "dynamicminmax05":    
        btc['ma1'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma1,0.5)
    elif tma1 == "dynamicminmax01":    
        btc['ma1'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma1,0.1)
    elif tma1 == "dynamicminmax075":    
        btc['ma1'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma1,0.75)
    elif tma1 == "dynamicminmax03":    
        btc['ma1'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma1,0.3)    
    elif tma1 == "dynamicminmax125":    
       btc['ma1'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma1,1.25)   
        
        
    if tma2 == "sma":
        btc['ma2'] = btc['close'].rolling(window=ma2).mean()
    elif tma2 == "ema":
        btc['ma2'] = btc['close'].ewm(span=ma2, adjust=False).mean()
    elif tma2 == "wma":
        weights = np.arange(1,ma2+1) #this creates an array with integers 1 to 10 included
        btc['ma2'] = btc['close'].rolling(ma2).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    elif tma2 == "dema":
        ema1 = btc['close'].ewm(span=ma2, adjust=False).mean()
        ema2 = ema1.ewm(span=ma2, adjust=False).mean()
        btc['ma2'] = 2 * ema1 - ema2            
    elif tma2 == "tema":
        btc['ma2'] = TA.TEMA(btc, ma2)
    elif tma2 == "trima":
        btc['ma2'] = TA.TRIMA(btc, ma2)     
    elif tma2 == "vama":
        btc['ma2'] = TA.VAMA(btc, ma2)              
    elif tma2 == "kama":
        btc['ma2'] = TA.KAMA(btc, er= 30, ema_fast = int(round(min(ma1,ma2))/10) , ema_slow =int(round(max(ma1,ma2))/10)  , period=40)   
    elif tma2 == "zlema":
        btc['ma2'] = TA.ZLEMA(btc, ma2)
    elif tma2 == "evwma":
        btc['ma2'] = TA.EVWMA(btc, ma2)
    elif tma2 == "clwma":
        btc['weights'] = btc['high'] -btc['low'] 
        btc['ma2'] = get_weighted_average(btc,ma2,"close","weights").values 
    elif tma2 == "qclwma":
        btc['weights'] = pow((btc['high'] -btc['low']) ,4).axis =1
        btc['ma2'] = get_weighted_average(btc,ma2,"close","weights").values   
    elif tma2 == "vwma":
        btc['ma2'] = get_weighted_average(btc,ma2,"close","volume").values
    elif tma2 == "3cvwma":
        btc['ma2'] = vqvwma(btc.close,btc.volume,ma2*m1,3)
    elif tma2 == "2cvwma":
        btc['ma2'] = vqvwma(btc.close,btc.volume,ma2*m1,2)         
    elif tma2 == "minmax_c":
        min2=btc['close'].rolling(ma2).min()
        max2=btc['close'].rolling(ma2).max()
        btc['ma2'] = (min2 +max2) / 2    

    elif tma2 == "minmax_o":
        min2=btc['open'].rolling(ma2).min()
        max2=btc['open'].rolling(ma2).max()
        btc['ma2'] = (min2 +max2) / 2  
        
    elif tma2 == "minmaxlh":
        min2=btc['low'].rolling(ma2).min()
        max2=btc['high'].rolling(ma2).max()
        btc['ma2'] = (min2 +max2) / 2        
        
    elif tma2 == "osciladorvol":
        btc['ma2'] = btc['oscilador'].ewm(span=ma2, adjust=False).mean()

    elif tma2 == "dynamicminmax1":    
        btc['ma2'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma2,1)
    elif tma2 == "dynamicminmax05":    
        btc['ma2'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma2,0.5)
    elif tma2 == "dynamicminmax01":    
        btc['ma2'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma2,0.1)
    elif tma1 == "dynamicminmax075":    
        btc['ma2'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma2,0.75)
    elif tma1 == "dynamicminmax03":    
        btc['ma2'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma2,0.3)    
    elif tma1 == "dynamicminmax125":    
       btc['ma2'] = dynamic_minmax( pd.Series.to_numpy(btc['close']), ma2,1.25)   
    return btc

def mas_trader(btcdataset,ma1,tma1,ma2,tma2, fee,tradeSize, purpose):                    
    # fee = 1 - (fee/100)   
    fee = (fee/100)   

    
    btc = copy.deepcopy(btcdataset)     

    
    btc=btc.dropna() 
    
    btc = moving_averages(btc,tma1,tma2,ma1,ma2)
    btc=btc.dropna() 
    
    
    btc.loc[(btc['ma2'] < btc['ma1']) & (btc['ma2'].shift(1) >= btc['ma1'].shift(1)), 'flag'] = 'Short'  
    btc.loc[(btc['ma2'] > btc['ma1']) & (btc['ma2'].shift(1) <= btc['ma1'].shift(1)), 'flag'] = 'Long'  

        
    btc=btc.dropna()  
    
    btc=btc.iloc[1:,]

    btc['test'] = btc['flag'] == btc['flag'].shift(1)     
    
    btc = btc.loc[btc['test']   == False, ]    

    sp=(tradeSize /btc['close'].shift(1))            
    btc['profit1'] =tradeSize -( sp * btc['close']) -(tradeSize*fee)-(sp*fee)

    btc['profit2'] = ((btc['close']- btc['close'].shift(1)) -  (btc['close']*fee) - (btc['close'].shift(1)*fee)) / btc['close'].shift(1) * tradeSize
    
    
    
    btc['profitlongs'] = btc.loc[ btc['flag']=='Short' , 'profit2']
    btc['profitshorts'] =  btc.loc[ btc['flag']=='Long' , 'profit1']
    btc['profitstotal'] = btc[['profitlongs', 'profitshorts']].sum(axis=1)
    btc['cumsumprofits'] = 1000+btc['profitstotal'].cumsum(skipna=True)
    btc['cummaxprofits'] = btc['cumsumprofits'].cummax(skipna=True)
    
    btc['drawdown'] = ( btc['cummaxprofits']  - btc['cumsumprofits']) /btc['cummaxprofits']
    bnh = (btc['close'].iat[-1]- btc['close'].iat[0]) / btc['close'].iat[0] * 1000

    

    if purpose == "loop":
        profitlongs =  round(np.nansum(btc['profitlongs']),1)
        profitshorts = round(np.nansum(btc['profitshorts']),1)    
        trades = len(btc)
        profitstotal = round(profitlongs+profitshorts,1)
        maxdrawdown= round(btc['drawdown'].max()*100,1)
        beats_bnh= np.where( bnh > (btc['cumsumprofits'].iat[-1]-1000) ,"No","Yes"  )



        output = {
                    "profitlongs"  : profitlongs,
                          "profitshorts" : profitshorts,
                          "profitstotal": profitstotal,
                          "trades" : trades,
                          "%_winning_trades": round(len(btc.loc[btc['profitstotal'] >0 ])/len(btc)*100,1),
                          "avg_profit": round(profitstotal/trades,1),
                          "max_drawdown": maxdrawdown,
                          "beats_bnh": beats_bnh
      }
        
        
        
    elif purpose == "standalone":
        output = btc
    return output


def loop_single_pair(btcdataset, mini1,maxi1,mini2,maxi2, tma1,tma2, rows,same_same):
    
    ma1 = np.random.uniform(mini1, maxi1, rows).round(0).astype(int)
    ma2 = np.random.uniform(mini2, maxi2, rows).round(0).astype(int)
    

    if type(tma1) == str:
        tma1 = [tma1]
    else:
        pass
        
    if type(tma2) == str:
        tma2 = [tma2]      
    else:
        pass
        
    if len(same_same) != 1:
        random.shuffle(tma1)
    else:
        pass
    
    tma1 = ( tma1 * rows)[0:rows]
    
    tma2 = ( tma2 * rows)[0:rows]
    
    grid = pd.DataFrame({
            'ma1':ma1,
            'ma2':ma2,
            'tma1':tma1,
            'tma2':tma2})
        
    
    lprofitlongs=[]
    lprofitshorts=[]
    lprofitstotal=[]
    ltrades=[]
    ltrades_positivos =[]
    lmaxdrawdown=[]
    lbeats_bnh = []

    
    for i in range(len(grid)):
        pma1 = grid.at[i,'ma1']  
        pma2 = grid.at[i,'ma2']
        ptma1 = grid.at[i,'tma1']
        ptma2 = grid.at[i,'tma2']    
            
        try:    
           output = mas_trader(btcdataset, pma1,ptma1,pma2,ptma2, 0.075,1000, "loop")
           print(i)
        except:
             output = {
                        "profitlongs"  : 0,
                              "profitshorts" : 0,
                              "profitstotal": 0,
                              "countTrue" : 0,
                              "trades" : 0,
                              "%_winning_trades": 0,
                              "max_drawdown":0,
                              "beats_bnh":""
                              }
            
            
        lprofitlongs.append(output['profitlongs'])
        lprofitshorts.append(output['profitshorts'])
        lprofitstotal.append(output['profitstotal'])
        ltrades.append(output['trades'])  
        ltrades_positivos.append(output['%_winning_trades'])  
        lmaxdrawdown.append(output['max_drawdown'])  
        lbeats_bnh.append(output['beats_bnh']) 
     
    grid['profitsshorts'] = lprofitshorts
    grid['profitslongs'] = lprofitlongs      
    grid['profitstotal'] = lprofitstotal
    grid['trades'] = ltrades
    grid['%_winning_trades'] = ltrades_positivos
    grid['avg_profit'] = round(grid['profitstotal']/ grid['trades'],1)
    grid['max_drawdown'] = lmaxdrawdown
    grid['beats_bnh'] = lbeats_bnh

    grid = grid.sort_values(by=['profitstotal'], ascending=False)
    grid = grid.reset_index(drop=True)
    
    grid = grid.loc[ grid.trades >1, ]
    
    grid = grid[0:10]

    return grid

def update_chart(df,pma1,tma1,pma2,tma2,ticker_input,logscale):    

    mkdata = moving_averages(df,tma1,tma2,pma1,pma2)
    
    output = mas_trader(mkdata,pma1,tma1,pma2,tma2, 0.075,1000, "standalone")    
    output['profitstotal'] = output.profitstotal.cumsum()
    profitstotal = output.loc[ output.profitstotal.notnull() , 'profitstotal']
    
    
    ventas  = output[ output['flag'] == "Short"]    
    compras = output[ output['flag'] == "Long" ]
    
        
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    
    fig3.add_trace(go.Scatter(x=mkdata.index , y=mkdata['close'],
                        mode='lines',
                        name= str(ticker_input)+' - close'))

    fig3.add_trace(go.Scatter(x=profitstotal.index, y=profitstotal,
                        mode='lines',
                        name='Accum. profits'), secondary_y = True)
    
    
    fig3.add_trace(go.Scatter(x=mkdata.index , y=mkdata['ma1'],
                        mode='lines',
                        name= str(tma1)+" "+str(pma1)  ))
    
    fig3.add_trace(go.Scatter(x=mkdata.index , y=mkdata['ma2'],
                        mode='lines',
                        name=str(tma2)+" "+str(pma2)))
    
    
    fig3.add_trace(go.Scatter(x=compras.index, y=compras['close'],
                        mode='markers',
                        name='buy',
                        marker=dict(
                color='Green',
                size=9,
                line=dict(
                    color='Black',
                    width=1
                ))))
    
    fig3.add_trace(go.Scatter(x=ventas.index, y=ventas['close'],
                        mode='markers',
                        name='sell',
                        marker=dict(
                color='Red',
                size=9,
                line=dict(
                    color='Black',
                    width=1
                ))))
    
    fig3.update_layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  # width=870,
                  # height=565,
                  hovermode='x',
                  autosize=True,
                  # title={'text': str(ticker_input)+" - Close", 'font': {'color': 'white'}, 'x': 0.5},
                  xaxis={'range': [mkdata.index.min(), mkdata.index.max()]},
                  # yaxis={'range': [mkdata.close.min(), mkdata.close.max()]},
                  transition={'duration': 400,'easing': 'linear-in-out' },
                  margin={'l':10, 'r':10, 't':10, 'b':10},
                  legend=dict(    orientation="h",    yanchor="bottom",    y=1.02,    xanchor="right",    x=1)
              ),    
        
    fig3.update_yaxes(title_text="Accumulated profits", secondary_y=True)
    fig3.update_yaxes(title_text="Price", secondary_y=False)

    if logscale==True:
        fig3.update_layout(    
                  yaxis={'type': 'log',                  
                         'range': [np.log10( mkdata.close.min() ) ,np.log10( mkdata.close.max()) ] })
    else:
        fig3.update_layout(    
                  yaxis={'type': 'linear',                  
                         'range': [( mkdata.close.min() ) ,( mkdata.close.max()) ] })        
    fig3.update_yaxes(showgrid=False, zeroline=True, secondary_y=True)
    
    return(fig3.data,fig3.layout)


def updatetable(df,pma1,tma1,pma2,tma2):    
    mkdata = moving_averages(df,tma1,tma2,pma1,pma2)
    output = mas_trader(mkdata,pma1,tma1,pma2,tma2, 0.075,1000, "loop")    
    return output

allvalues =['sma','ema','wma','dema','tema','trima','vama','zlema','minmax_c']

listamedias =     [{'label': 'SMA', 'value': 'sma'},
                   {'label': 'EMA', 'value': 'ema'},
                   {'label': 'WMA', 'value': 'wma'},
                   {'label': 'DEMA', 'value': 'dema'},
                   {'label': 'TEMA', 'value': 'tema'},
                   {'label': 'TRIMA', 'value': 'trima'},
                   {'label': 'VAMA', 'value': 'vama'},
                   {'label': 'ZLEMA', 'value': 'zlema'},
                    # {'label': 'EVWMA', 'value': 'evwma'},
                   {'label': 'MINMAX/2', 'value': 'minmax_c'}]

intervals =       [{'label': '1D', 'value': '1d'},
                   {'label': '1H', 'value': '60m'}]

osma1 = 328
osma2 = 311
ostma1 = "tema"
ostma2 = "tema"
osticker='ggal'
ostf ='60m'
p="2y"

stock = yf.Ticker(osticker )
    
# get historical market data
mkdata = stock.history(period=p,interval = ostf)
mkdata = mkdata.rename(columns={"Close": "close", "Volume": "volume"})
osdata = updatetable(mkdata,osma1,ostma1,osma2,ostma2)
osdata = pd.DataFrame([osdata])
osdata.insert(loc=0, column='tma2', value=ostma2)
osdata.insert(loc=0, column='tma1', value=ostma1)
osdata.insert(loc=0, column='ma2', value=osma2)
osdata.insert(loc=0, column='ma1', value=osma1)




ostrace,oslayout=update_chart(mkdata,osma1,ostma1,osma2,ostma2,osticker,False)


server = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server.config.suppress_callback_exceptions = False

server_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

server.layout = html.Div(
    [
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Moving Averages Crossover Strategy Scanner", className="app__header__title"),
                        html.P(
                            "This app enables quick scanning of random moving averages crossover strategies for swing trading.",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
                html.Div(
                    [
                        html.Img(
                            src=server.get_asset_url("dash-new-logo.png"),
                            className="app__menu__img",
                        )
                    ],
                    className="app__header__logo",
                ),
            ],
            className="app__header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.H6("Strategy Chart", className="graph__title")]
                        ),
                        dcc.Graph(
                            id="chart",
                            figure=dict(data=ostrace,
                                layout=oslayout
                            ),
                        ),
                        html.Div(
                        [
                              dash_table.DataTable(
                                            id='singletable',
                                            
                                            columns=[{"name": i, "id": i} for i in osdata.columns],
                                            
                                            data=osdata.to_dict('records'),                                                
                                               style_table={'width': '80%', 'padding-left':'4%','padding-bottom':'2%'},
                                               style_as_list_view=False,
                                               style_header={'backgroundColor': '#061E44'},
                                               style_cell={
                                                        'backgroundColor': 'rgba(0,0, 0, 0)',
                                                        'color': 'white'
                                                    },
                                        )   ], className = "summary__table" )
                    ],
                    className="two-thirds column wind__speed__container",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Custom",
                                            className="graph__subtitle",
                                        )
                                    ]
                                ),

                            html.Div( 
                                    [html.P("Ticker:",style={'width': '20%', 'display': 'inline-block'}), 
                                     
                                     
                                     dcc.Input( id="ticker_input",  placeholder='Ingrese un ticker...',
                                                type='text',
                                                value=osticker,
                                                style={'width': '22%', 'display': 'inline-block'}),

                                       html.Div(    [        
                                     dcc.Dropdown(
                                            id='intvl1',
                                            options= intervals,
                                            value=ostf, multi = False,  clearable=False)],style={'width': '15%', 'display': 'inline-block', 'margin-left':'5%',  'vertical-align': 'middle'}),
   
                                    dcc.Checklist(id="log_checkbox",
                                            options=[{'label': 'Log', 'value': "log"}],
                                            value=[],
                                            style={'width': '15%', 'display': 'inline-block', 'margin-left':'5%'}),
                                        ], 
                                    className = "horizontal__container"
                                ),
                            html.Div( [
                                        html.P("Date range:",style={'width': '20%', 'display': 'inline-block',  'vertical-align': 'bottom'}), 
                                       html.Div(
                                    [  
                                        dcc.RangeSlider(
                                                id='date_range_slider',
                                                min=0,
                                                max=100,
                                                step=1,
                                                value=[0, 100],
                                                allowCross=False,
                                                marks={0: "Start", 100: "End"},
                                                tooltip = { 'always_visible': False }
                                            )
                                     ],
                                    className="slider",style={'width': '70%', 'display': 'inline-block', 'margin-left':'5%',  'vertical-align': 'middle'}) ], 
                                    className = "horizontal__container"
                                ),                                    
                        html.Div( [
                            html.P("M. Avg. 1:",style={'width': '20%', 'display': 'inline-block','margin-top': '15px', 'vertical-align': 'top'}),                                      
                                html.Div( 
                                    [ 
                                dcc.Dropdown(
                                                id='tma1',
                                                options=listamedias,
                                                value=ostma1,
                                                clearable=False,
                                                className = "select__input"
                                            ) 
                                    ]
                                    , className = "select__input"
                                    , style={'width': '20%', 'display': 'inline-block', 'margin-left':'0%'}
                                ),
                                html.Div(
                                    [   dcc.Slider(
                                                    id='pma1',
                                                    min=1,
                                                    max=500,
                                                    step=1,
                                                    value=osma1,
                                                    tooltip = { 'always_visible': False },
                                                    marks = {1: "1", 500: "500"},
                                                )
                                        
                                     ],
                                    className="slider",
                                            style={'width': '50%', 'display': 'inline-block', 'margin-left':'5%'} )  ], 
                                                
                                    className = "horizontal__container"), 
                        
                        html.Div( [ 
                            html.P("M. Avg. 2:",style={'width': '20%', 'display': 'inline-block','margin-top': '15px', 'vertical-align': 'top'}),                                                             
                        
                                html.Div(
                                    [   
                                    dcc.Dropdown(
                                                id='tma2',
                                                options=listamedias,
                                                value=ostma2,  
                                                clearable=False,
                                                className = "select__input"
                                            )
                                        
                                     ], className = "select__input", 
                                style={'width': '20%', 'display': 'inline-block', 'margin-left':'0%'}),
                                html.Div(
                                    [   dcc.Slider(
                                                    id='pma2',
                                                    min=1,
                                                    max=500,
                                                    step=1,
                                                    value=osma2,
                                                    tooltip = { 'always_visible': False },
                                                    marks={1: "1", 500: "500"},
                                                )
                                     ],
                                    className="slider",
                                    style={'width': '50%', 'display': 'inline-block', 'margin-left':'5%'}),
                                
                                ], 
                                    className = "horizontal__container"), 
                            ],
                            className="graph__container first",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Random Scan", className="graph__subtitle"
                                        )
                                    ]
                                ),
                                html.Div(
                                    [  
                                     dcc.Dropdown(
                                            id='tma1_2',
                                            options= listamedias,
                                            multi = True,  
                                            clearable=False,
                                            value=allvalues
                                        )
                                        
                                     ], className = "select__input"),                                    
                                html.Div(
                                    [  
                                            dcc.RangeSlider(
                                                id ='my-range-slider_1',
                                                min = 1,
                                                max = 500,
                                                step = 1,
                                                value = [1, 500],
                                                allowCross = False,
                                                marks = {1: "1", 500: "500"},
                                                tooltip = { 'always_visible': False }
                                            )
                                     ],
                                    className="slider"),                                    
                                html.Div(
                                    [  
                                    dcc.Dropdown(
                                            id='tma2_2',
                                            options= listamedias,
                                            value= allvalues, multi = True,  clearable=False
                                        )
                                        
                                     ], className = "select__input"),                                    
                                html.Div(
                                    [  
                                    dcc.RangeSlider(
                                                id='my-range-slider_2',
                                                min=1,
                                                max=500,
                                                step=1,
                                                value=[1, 500],
                                                allowCross=False,
                                                marks={1: "1", 500: "500"},
                                                tooltip = { 'always_visible': False }
                                            )
                                     ],
                                    className="slider"), 
                                html.Div(
                                    [  
                                    html.Button('Scan', id = 'submit-val button'),
                                                       dcc.Store(id='memory-output'),
                                                       dcc.Store(id='memory-output2')
                                     ]),  
                                html.Div(
                                    [  dcc.Checklist(id="check_all",
                                            options=[{'label': 'Check all MAs', 'value': "Yes"}],
                                            value=["Yes"]) ], id='checklist-container'),
                                html.Div(
                                    [  dcc.Checklist(id="same_same",
                                            options=[{'label': 'Use identical pairs', 'value': "Yes"}],
                                            value=["Yes"]) ])
                            ],
                            className="graph__container second",
                        ),
                    ],
                    className="one-third column histogram__direction",
                ),
                
            ],
            className="app__content",
        ),
        html.Div(
    [
   html.Div(
                    [
                        html.Div(
                            [html.H6("Top 10 random strategies:", 
                                     className="graph__title")]
                        ),
                        
                        html.Div(
                            [
                                  dash_table.DataTable(
                                                id='table',
                                                columns=[{"name": i, "id": i} for i in osdata.columns],
                                                row_selectable = 'single',
                                                   style_table={'width': '80%', 'padding-left':'3%'},
                                                   style_as_list_view=False,
                                                   style_header={'backgroundColor': '#061E44'},
                                                   style_cell={
                                                            'backgroundColor': 'rgba(0,0, 0, 0)',
                                                            'color': 'white'
                                                        },
                                            )   ], className = "summary__table" ),
                    ],
                    className="two-thirds column wind__speed__container",
                ),
             html.Div(
                     [        html.Div(
                             [
                                                         html.Div(
                            [html.H6("Create Telegram Notification:", className="graph__subtitle")]
                        ),
                                 ],
                            className="graph__container first",
                        ),
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)


@server.callback(
    Output('memory-output', 'data'),
    [Input('ticker_input', 'value'),
    Input('intvl1', 'value')],prevent_initial_call=False  # whatever this will be
)
def generate_df_callback(ticker_input,intvl1):    
    if intvl1 != "1d":
        p = "2y"
    else:
        p = "max"
    stock = yf.Ticker( ticker_input )
    mkdata = stock.history(period=p,interval = intvl1)
    mkdata = mkdata.rename(columns={"Close": "close", "Volume": "volume"})
    mkdata = mkdata.to_json()
    return mkdata


@server.callback(
    [Output('memory-output2', 'data',allow_duplicate=True),    
     Output('table', 'data',allow_duplicate=True)],
    [Input('memory-output', 'data'),
     Input('date_range_slider', 'value')],prevent_initial_call=True  # whatever this will be
)
def update_date_range(data,value):    
    start = value[0]
    end = value[1]            
    mkdata = pd.read_json(data)    
    tdelta = mkdata.index.max() - mkdata.index.min()
    nstart = mkdata.index.min() + (tdelta * (start/100))
    nend = mkdata.index.min()  + (tdelta * (end/100))
    mkdata = mkdata[(mkdata.index>= nstart) & (mkdata.index<= nend) ]
    mkdata = mkdata.to_json()
    return mkdata, np.nan

@server.callback(
    [Output('chart', 'figure'),
    Output('singletable', 'data',allow_duplicate=True),    
    Output('singletable', 'columns'),
    Output("singletable", "selected_rows",allow_duplicate=True)],
    [Input('pma1', 'value'),
    Input('tma1', 'value'),
    Input('pma2', 'value'),
    Input('tma2', 'value'),
    Input('memory-output2', 'data'),
    Input('log_checkbox', 'value') ],
    State('ticker_input', 'value') ,prevent_initial_call=True)
def updateChart(pma1,tma1,pma2,tma2,df,log,ticker_input):   

    df = pd.read_json(df)
    
    if len(log) == 1:
        logscale = True
    else:
        logscale = False        

    trace,layout=update_chart(df,pma1,tma1,pma2,tma2,ticker_input,logscale)
    
    graphdata =  {
        'data': trace,
        'layout': layout
    }
    
    data2 = updatetable(df,pma1,tma1,pma2,tma2)
    
    data2 = pd.DataFrame([data2])

    data2.insert(loc=0, column='tma2', value=tma2)
    data2.insert(loc=0, column='tma1', value=tma1)
    data2.insert(loc=0, column='ma2', value=pma2)
    data2.insert(loc=0, column='ma1', value=pma1)
    
    columns= [{'name': col, 'id': col} for col in data2.columns ]
    
    data2= data2.to_dict('records') 
        
    selected = [0]
        
    return graphdata, data2, columns,selected

@server.callback(
    [Output('table', 'data'),    
    Output('table', 'columns'),
    Output("table", "selected_rows",allow_duplicate=True)],
     Input('submit-val button','n_clicks'),
    [State('memory-output2', 'data'),
     State('my-range-slider_1', 'value'),
    State('my-range-slider_2', 'value'),
    State('tma1_2', 'value'),
    State('tma2_2', 'value'),
    State('same_same', 'value') ], prevent_initial_call=True)

def updateTable(n_clicks, data,slider_1, slider_2,tma1_2,tma2_2,same_same):
    ctx = dash.callback_context

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    selected = [0]
    
    if button_id == "submit-val button":
        
        mini_1 = slider_1[0]
        maxi_1 = slider_1[1]
        
        mini_2 = slider_2[0]
        maxi_2 = slider_2[1]
        
        df = pd.read_json(data)    
        
        grid = loop_single_pair(df, mini_1,maxi_1,mini_2,maxi_2, tma1_2,tma2_2, 20,same_same)
        
        columns= [{'name': col, 'id': col} for col in grid.columns]
        
        data2= grid.to_dict('records')  
    else:
        pass
 
    
    return data2, columns,selected        

@server.callback(
    [Output('pma1', 'value'),
     Output('pma2', 'value'),
     Output('tma1', 'value'),
     Output('tma2', 'value'),],    
      Input('table', 'selected_rows') ,      
      [State('table', 'data') ]
      ,prevent_initial_call=True)
def updateControls(selected_rows,data):   
    ix=selected_rows[0]
    pma1 = data[ix]['ma1']
    pma2 = data[ix]['ma2']
    tma1 = data[ix]['tma1']
    tma2 = data[ix]['tma2']
    
    return pma1,pma2,tma1,tma2

@server.callback(
    [Output('tma1_2', 'value'),
    Output('tma2_2', 'value')],
    [Input('check_all', 'value')],prevent_initial_call=True)
def updateDropdowns(value):    
    if len(value)==1:
        return allvalues,allvalues
    else:
        raise PreventUpdate()

@server.callback(
    Output('checklist-container', 'children'),
    [Input('tma1_2', 'value'),
    Input('tma2_2', 'value')],prevent_initial_call=False)
def updateCheckbox(value,value2):    
    if len(value)+len(value2) == len(allvalues)*2:
        raise PreventUpdate()
    else:
        return dcc.Checklist(id="check_all",
                                            options=[{'label': 'Check all MAs', 'value': "Yes"}],
                                            value=[])

if __name__ == "__main__":
    server.run_server(debug=True)









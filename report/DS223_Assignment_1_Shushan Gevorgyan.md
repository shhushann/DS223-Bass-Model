# DS 223, Assignment #1
# Bass Model
# *Shushan Gevorgyan*

## Libraries and Packages


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from scipy.optimize import least_squares
import plotly.express as px
import kaleido
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display
import numpy as np

```

## Time Innovation: *ThredUp AI Search* 
## Similar Product: *Vinted* 


```python
display(Image(filename=r'img/header.png'))
```


    
![png](DS223_Assignment_1_Shushan%20Gevorgyan_files/DS223_Assignment_1_Shushan%20Gevorgyan_4_0.png)
    


A past innovation that resembles ThredUp AI Search is Vinted, the peer-to-peer online marketplace for second-hand clothing that has been popular in Europe since the late 2000s. Both platforms aim to simplify and accelerate the process of buying and selling pre-owned clothing online. Vinted allowed users to list items, browse through curated categories, and find products through text-based searches and filters. Functionally, Vinted pioneered the concept of accessible, large-scale second-hand shopping online, connecting buyers and sellers in a user-friendly interface and encouraging sustainable fashion practices.

ThredUp AI Search builds upon this concept with the use of modern artificial intelligence and computer vision. Unlike Vinted’s primarily keyword- or category-based search, ThredUp lets users input ultra-specific phrases or upload images to find visually similar clothing items from millions of listings. This reduces guesswork and increases discoverability, enabling users who may be unfamiliar with brands or styles to shop sustainably with ease. While both innovations have expanded the second-hand fashion market, ThredUp’s AI-driven approach represents a technological evolution, improving user experience and driving higher engagement, as seen in its reported 38% year-over-year increase in searches per session.

## Data extraction

For this analysis, I sourced historical data on Vinted from Statista. The original data was provided in PPTX format as plots within a presentation, which required manual extraction. I was able to convert the visual data into Excel files for further processing. However, the Excel files I found only contained data up to 2021, which was insufficient for modeling the diffusion of the innovation. To address this, I combined the extracted historical data with more recent publicly available statistics to construct a complete time series suitable for Bass model estimation and forecasting.
I was also able to find data showing downloads of Vinted in 2024 by countires, which supported my answer for question N *6*. 

My main variable for the Bass model analysis is Gross Merchandise Volume (GMV) of Vinted worldwide from 2016 to 2024, measured in million USD. I also collected Revenue data, which served as an additional reference to validate the Bass model’s predictive function. GMV was chosen as the primary variable because it reflects the total value of all transactions on the platform, capturing the overall scale, adoption, and market activity more directly than revenue alone. Revenue, while important for financial performance, depends on commission rates and business model specifics, which can fluctuate independently of user adoption. Therefore, GMV provides a better proxy for the diffusion and popularity of the platform across users, making it ideal for modeling adoption patterns using the Bass diffusion model.

## Loading Data


```python
gmv_path = 'data/GMV Vinted 2016-2024 .xlsx'
revenue_path = 'data/Revenue Vinted 2017-2024.xlsx'
downloads_by_country_path = 'data/Downloads by Country.xlsx'
gmv_df = pd.read_excel(gmv_path)       
revenue_df = pd.read_excel(revenue_path)
downloads = pd.read_excel(downloads_by_country_path)

```


```python
gmv_df.columns = gmv_df.columns.str.strip()
revenue_df.columns = revenue_df.columns.str.strip()

print(gmv_df)
print(revenue_df)
```

       Year      GMV
    0  2016     29.5
    1  2017    114.5
    2  2018    506.6
    3  2019   1154.3
    4  2020   2424.3
    5  2021   4829.5
    6  2022   6487.2
    7  2023  10720.0
    8  2024  12564.9
       Year  Revenue
    0  2017     10.0
    1  2018     30.0
    2  2019     84.0
    3  2020    150.0
    4  2021    245.3
    5  2022    370.2
    6  2023    596.3
    7  2024    813.4



```python
full_data = pd.merge(gmv_df, revenue_df, on='Year', how='inner')  

print(full_data)
```

       Year      GMV  Revenue
    0  2017    114.5     10.0
    1  2018    506.6     30.0
    2  2019   1154.3     84.0
    3  2020   2424.3    150.0
    4  2021   4829.5    245.3
    5  2022   6487.2    370.2
    6  2023  10720.0    596.3
    7  2024  12564.9    813.4


## Estimating the parameters for Bass Model


```python
years = full_data['Year'].values
gmv = full_data['GMV'].values
rev = full_data['Revenue'].values
t = np.arange(len(years))
```


```python
def bass_model(params, t, actual):
    p, q, M = params
    Y = np.zeros(len(t))
    S = np.zeros(len(t))
    for i in range(len(t)):
        if i == 0:
            S[i] = min(actual[0], M)
            Y[i] = S[i]
        else:
            S[i] = (p + q * Y[i-1]/M) * (M - Y[i-1])
            Y[i] = Y[i-1] + S[i]
    return S
```


```python
def residuals(params, t, actual):
    return actual - bass_model(params, t, actual)
```


```python
initial_params = [0.03, 0.4, max(gmv)*2]  
bounds = ([0, 0, max(gmv)], [1, 1, 1e6])

```

## Bass Model Parameteres based on GMV


```python
result_gmv = least_squares(residuals, initial_params, bounds=bounds, args=(t, gmv))
p_hat_gmv, q_hat_gmv, M_hat_gmv = result_gmv.x


print(f"Estimated parameters(GMV):\np = {p_hat_gmv:.4f}\nq = {q_hat_gmv:.4f}\nM = {M_hat_gmv:.0f}")
```

    Estimated parameters(GMV):
    p = 0.0109
    q = 0.8901
    M = 55352


## Bass Model Parameteres based on Revenue


```python
result_rev = least_squares(residuals, initial_params, bounds = bounds, args=(t, rev))
p_hat_rev, q_hat_rev, M_hat_rev = result_rev.x
print(f"Estimated parameters(Revenue):\np = {p_hat_rev:.4f}\nq = {q_hat_rev:.4f}\nM = {M_hat_rev:.0f}")
```

    Estimated parameters(Revenue):
    p = 0.0044
    q = 0.5927
    M = 12565


## Forecast VS Real data

## Plotting GMV model


```python
S_pred = bass_model([p_hat_gmv, q_hat_gmv, M_hat_gmv], t, gmv)
plt.figure(figsize=(8,5))
plt.plot(years, gmv, 'o-', label='Actual GMV')
plt.plot(years, S_pred, 's--', label='Bass Model Prediction')
plt.xlabel('Year')
plt.ylabel('GMV (million USD)')
plt.title('Bass Model Fit to GMV')
plt.legend()
plt.grid(True)
plt.savefig('img/bass_model_fit_gmv.png', dpi=300)
plt.show()
```


    
![png](DS223_Assignment_1_Shushan%20Gevorgyan_files/DS223_Assignment_1_Shushan%20Gevorgyan_24_0.png)
    


## Plotting Revenue model


```python
S_pred = bass_model([p_hat_rev, q_hat_rev, M_hat_rev], t, rev)
plt.figure(figsize=(8,5))
plt.plot(years, rev, 'o-', label='Actual GMV')
plt.plot(years, S_pred, 's--', label='Bass Model Prediction')
plt.xlabel('Year')
plt.ylabel('Revenue (million USD)')
plt.title('Bass Model Fit to Revenue')
plt.legend()
plt.grid(True)
plt.savefig('img/bass_model_fit_rev.png', dpi=300)
plt.show()
```


    
![png](DS223_Assignment_1_Shushan%20Gevorgyan_files/DS223_Assignment_1_Shushan%20Gevorgyan_26_0.png)
    


## Future Forecasting for ThredUp AI Search


```python

future_years = np.arange(2024, 2035)
T = len(future_years)

S_future = np.zeros(T)
Y_future = np.zeros(T)

for i in range(T):
    if i == 0:
        S_future[i] = p_hat_gmv * M_hat_gmv
        Y_future[i] = S_future[i]
    else:
        S_future[i] = (p_hat_gmv + q_hat_gmv * (Y_future[i-1] / M_hat_gmv)) * (M_hat_gmv - Y_future[i-1])
        Y_future[i] = Y_future[i-1] + S_future[i]

df_pred = pd.DataFrame({
    'Year': future_years,
    'New_GMV_million_USD': S_future.round(1),
    'Cumulative_GMV_million_USD': Y_future.round(1),
    'Cumulative_pct_of_M': (Y_future / M_hat_gmv * 100).round(1)
})

print(df_pred)

```

        Year  New_GMV_million_USD  Cumulative_GMV_million_USD  Cumulative_pct_of_M
    0   2024                600.9                       600.9                  1.1
    1   2025               1123.4                      1724.2                  3.1
    2   2026               2069.1                      3793.3                  6.9
    3   2027               3704.7                      7497.9                 13.5
    4   2028               6289.3                     13787.3                 24.9
    5   2029               9666.4                     23453.7                 42.4
    6   2030              12376.7                     35830.3                 64.7
    7   2031              11459.7                     47290.0                 85.4
    8   2032               6218.1                     53508.1                 96.7
    9   2033               1606.3                     55114.4                 99.6
    10  2034                212.9                     55327.3                100.0



```python
plt.figure(figsize=(10,6))

plt.plot(df_pred['Year'], df_pred['New_GMV_million_USD'], marker='o', linestyle='-', color='blue', label='New GMV per Year (million USD)')

plt.plot(df_pred['Year'], df_pred['Cumulative_GMV_million_USD'], marker='s', linestyle='--', color='green', label='Cumulative GMV (million USD)')

plt.xlabel('Year')
plt.ylabel('GMV (million USD)')
plt.title('Bass Model Prediction for ThredUp AI Search (S-shaped Diffusion)')
plt.legend()
plt.grid(True)
plt.savefig('img/thredup_difussion.png', dpi=300)
plt.show()

```


    
![png](DS223_Assignment_1_Shushan%20Gevorgyan_files/DS223_Assignment_1_Shushan%20Gevorgyan_29_0.png)
    


## Global or Country-specific


```python
downloads
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Downloads</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>United Kingdom</td>
      <td>6.37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>France</td>
      <td>3.10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Italy</td>
      <td>3.09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>2.18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Poland</td>
      <td>2.14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spain</td>
      <td>2.11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Romania</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sweden</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Netherlands</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Greece</td>
      <td>0.97</td>
    </tr>
  </tbody>
</table>
</div>




```python
downloads = downloads.dropna()
downloads.columns = downloads.columns.str.strip()
downloads
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Downloads</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>United Kingdom</td>
      <td>6.37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>France</td>
      <td>3.10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Italy</td>
      <td>3.09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>2.18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Poland</td>
      <td>2.14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spain</td>
      <td>2.11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Romania</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sweden</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Netherlands</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Greece</td>
      <td>0.97</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.choropleth(
    downloads,
    locations="Country",
    locationmode="country names",
    color="Downloads",
    color_continuous_scale="Reds",  
    range_color=[0, downloads["Downloads"].max()],  
    title="Vinted App Downloads in 2024 by Country (millions)"
)


fig.update_geos(
    scope="europe",
    fitbounds="locations",
    visible=False
)

fig.write_html('img/vinted_map.html') 

fig.show()
```


<div>                            <div id="2b1b496a-600a-4081-b991-429190672eda" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("2b1b496a-600a-4081-b991-429190672eda")) {                    Plotly.newPlot(                        "2b1b496a-600a-4081-b991-429190672eda",                        [{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Country=%{location}<br>Downloads=%{z}<extra></extra>","locationmode":"country names","locations":["United Kingdom","France","Italy","Germany","Poland","Spain","Romania","Sweden","Netherlands","Greece"],"name":"","z":[6.37,3.1,3.09,2.18,2.14,2.11,1.26,1.19,0.97,0.97],"type":"choropleth"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"geo":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{},"scope":"europe","fitbounds":"locations","visible":false},"coloraxis":{"colorbar":{"title":{"text":"Downloads"}},"colorscale":[[0.0,"rgb(255,245,240)"],[0.125,"rgb(254,224,210)"],[0.25,"rgb(252,187,161)"],[0.375,"rgb(252,146,114)"],[0.5,"rgb(251,106,74)"],[0.625,"rgb(239,59,44)"],[0.75,"rgb(203,24,29)"],[0.875,"rgb(165,15,21)"],[1.0,"rgb(103,0,13)"]],"cmin":0,"cmax":6.37},"legend":{"tracegroupgap":0},"title":{"text":"Vinted App Downloads in 2024 by Country (millions)"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('2b1b496a-600a-4081-b991-429190672eda');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


In 2024, Vinted app downloads were substantial across multiple countries, with the United Kingdom leading at 6.37 million downloads, followed by France (3.10 million), Italy (3.09 million), and Germany (2.18 million). Other countries such as Poland, Spain, Romania, Sweden, the Netherlands, and Greece collectively contributed millions more, reflecting strong international adoption. This distribution demonstrates that the secondhand marketplace is not confined to a single country but has significant usage across Europe. Consequently, analyzing the diffusion of innovations like ThredUp AI Search on a global scale is appropriate, as it captures the broad market potential and network effects evident in similar international platforms.

## GMV to estimated adopters


```python
avg_gmv_per_user = 0.0001  #as gmv is in million USD

df_pred['New_Adopters'] = (df_pred['New_GMV_million_USD'] / avg_gmv_per_user).round(0)
df_pred['Cumulative_Adopters'] = (df_pred['Cumulative_GMV_million_USD'] / avg_gmv_per_user).round(0)

print(df_pred[['Year','New_Adopters','Cumulative_Adopters']])
```

        Year  New_Adopters  Cumulative_Adopters
    0   2024     6009000.0            6009000.0
    1   2025    11234000.0           17242000.0
    2   2026    20691000.0           37933000.0
    3   2027    37047000.0           74979000.0
    4   2028    62893000.0          137873000.0
    5   2029    96664000.0          234537000.0
    6   2030   123767000.0          358303000.0
    7   2031   114597000.0          472900000.0
    8   2032    62181000.0          535081000.0
    9   2033    16063000.0          551144000.0
    10  2034     2129000.0          553273000.0



```python
fig, ax1 = plt.subplots(figsize=(10,6))

ax1.bar(df_pred['Year'], df_pred['New_Adopters'], color='skyblue', label='New Adopters per Year')
ax1.set_xlabel('Year')
ax1.set_ylabel('New Adopters (number of users)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(df_pred['Year'], df_pred['Cumulative_Adopters'], marker='o', linestyle='--', color='red', label='Cumulative Adopters')
ax2.set_ylabel('Cumulative Adopters', color='red')
ax2.tick_params(axis='y', labelcolor='red')


lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('Bass Model Prediction for ThredUp AI Search – Adoption Over Time')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.savefig('img/thredup_gmv_forecast.png', dpi=300)

plt.show()

```


    
![png](DS223_Assignment_1_Shushan%20Gevorgyan_files/DS223_Assignment_1_Shushan%20Gevorgyan_37_0.png)
    


# Summary

Using Vinted as a look-alike innovation, we estimated the Bass diffusion model parameters for the marketplace GMV: p = 0.0109 (coefficient of innovation), q = 0.8901 (coefficient of imitation), and M = 55,352 million USD (market potential). These parameters indicate that adoption is heavily driven by social contagion and imitation, consistent with peer-to-peer marketplaces where word-of-mouth and network effects play a major role.

Applying these parameters to ThredUp AI Search, we forecast the GMV growth over the next decade. The model predicts a gradual start in 2024 with 600.9 million USD in new GMV, accelerating rapidly as the technology spreads: by 2027, cumulative GMV reaches 7,498 million USD (13.5% of market potential), and by 2030, adoption crosses 64.7% of M. Peak adoption occurs around 2032–2033, after which growth slows, approaching saturation at the total market potential of 55,352 million USD by 2034.

This diffusion path reflects a typical S-shaped adoption curve: slow initial uptake due to early adopters experimenting with AI-based second-hand shopping, followed by rapid growth as the tool gains awareness, and eventual plateau as most of the target market has adopted the innovation. The model highlights the potential for ThredUp AI Search to significantly accelerate second-hand fashion adoption, leveraging the same network-driven dynamics that fueled Vinted’s success.

### References

1. **Statista.** (2024). *Vinted: Study Overview*. Retrieved from [https://www.statista.com/study/172216/vinted/](https://www.statista.com/study/172216/vinted/)

2. **Statista.** (2024). *Vinted App Downloads by Country*. Retrieved from [https://www.statista.com/statistics/1447603/vinted-app-downloads-by-country/](https://www.statista.com/statistics/1447603/vinted-app-downloads-by-country/)

3. **Time.** (2024). *Easier Secondhand Shopping: ThredUp AI Search*. Retrieved from [https://time.com/7094866/thredup-ai-search/](https://time.com/7094866/thredup-ai-search/)

4. **Vinted.** (2024). *How It Works*. Retrieved from [https://www.vinted.com/how_it_works](https://www.vinted.com/how_it_works)

5. **Course Slides.** (2024). *DS-223: Bass Model*. [PDF file]

6. **GeeksforGeeks.** (2025). *Bass Diffusion Model*. Retrieved from [https://www.geeksforgeeks.org/machine-learning/bass-diffusion-model/](https://www.geeksforgeeks.org/machine-learning/bass-diffusion-model/)


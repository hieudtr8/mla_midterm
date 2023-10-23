# # Summary Stats
# AAPL.describe()

# # General info
# AAPL.info()

# Let's see a historical view of the closing price
# plt.figure(figsize=(15, 10))
# plt.subplots_adjust(top=1.25, bottom=1.2)

# for i, company in enumerate(company_list, 1):
#     plt.subplot(2, 2, i)
#     company['Adj Close'].plot()
#     plt.ylabel('Adj Close')
#     plt.xlabel(None)
#     plt.title(f"Closing Price of {tech_list[i - 1]}")
    
# plt.tight_layout()
# plt.show()

# ma_day = [10, 20, 50]

# for ma in ma_day:
#     for company in company_list:
#         column_name = f"MA for {ma} days"
#         company[column_name] = company['Adj Close'].rolling(ma).mean()
        

# fig1, axes = plt.subplots(nrows=2, ncols=2)
# fig1.set_figheight(10)
# fig1.set_figwidth(15)

# AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
# axes[0,0].set_title('APPLE')

# GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
# axes[0,1].set_title('GOOGLE')

# MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
# axes[1,0].set_title('MICROSOFT')

# AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
# axes[1,1].set_title('AMAZON')

# fig1.tight_layout()
# fig.show()

# We'll use pct_change to find the percent change for each day
# for company in company_list:
#     company['Daily Return'] = company['Adj Close'].pct_change()

# # Then we'll plot the daily return percentage
# fig, axes = plt.subplots(nrows=2, ncols=2)
# fig.set_figheight(10)
# fig.set_figwidth(15)

# AAPL['Daily Return'].plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
# axes[0,0].set_title('APPLE')

# GOOG['Daily Return'].plot(ax=axes[0,1], legend=True, linestyle='--', marker='o')
# axes[0,1].set_title('GOOGLE')

# MSFT['Daily Return'].plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
# axes[1,0].set_title('MICROSOFT')

# AMZN['Daily Return'].plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
# axes[1,1].set_title('AMAZON')

# fig.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 9)) 

# for i, company in enumerate(company_list, 1):
#     plt.subplot(2, 2, i)
#     company['Daily Return'].hist(bins=50)
#     plt.xlabel('Daily Return')
#     plt.ylabel('Counts')
#     plt.title(f'{company_name[i - 1]}')
    
# plt.tight_layout()
# plt.show()

# Grab all the closing prices for the tech stock list into one DataFrame

# closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']

# # Make a new tech returns DataFrame
# tech_rets = closing_df.pct_change()
# tech_rets.head()


# sns.jointplot(x='GOOG', y='GOOG', data=tech_rets, kind='scatter', color='seagreen')
# sns.jointplot(x='GOOG', y='MSFT', data=tech_rets, kind='scatter')
# sns.pairplot(tech_rets, kind='reg')

# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
# return_fig = sns.PairGrid(tech_rets.dropna())

# # Using map_upper we can specify what the upper triangle will look like.
# return_fig.map_upper(plt.scatter, color='purple')

# # We can also define the lower triangle in the figure, inclufing the plot type (kde) 
# # or the color map (BluePurple)
# return_fig.map_lower(sns.kdeplot, cmap='cool_d')

# # Finally we'll define the diagonal as a series of histogram plots of the daily return
# return_fig.map_diag(plt.hist, bins=30)


# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
# returns_fig = sns.PairGrid(closing_df)

# # Using map_upper we can specify what the upper triangle will look like.
# returns_fig.map_upper(plt.scatter,color='purple')

# # We can also define the lower triangle in the figure, inclufing the plot type (kde) or the color map (BluePurple)
# returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

# # Finally we'll define the diagonal as a series of histogram plots of the daily return
# returns_fig.map_diag(plt.hist,bins=30)

# plt.figure(figsize=(12, 10))

# plt.subplot(2, 2, 1)
# sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
# plt.title('Correlation of stock return')

# plt.subplot(2, 2, 2)
# sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
# plt.title('Correlation of stock closing price')

# rets = tech_rets.dropna()

# area = np.pi * 20

# plt.figure(figsize=(10, 8))
# plt.scatter(rets.mean(), rets.std(), s=area)
# plt.xlabel('Expected return')
# plt.ylabel('Risk')

# for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
#     plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
#                  arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
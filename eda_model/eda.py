import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

df = pd.read_csv('data/df.csv')

save_dir = 'crisp-dm/eda_visualizations/'
def perform_eda(df, save=False):
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    print("Data Types:")
    print(df.dtypes)
    print("\n")

    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / df.shape[0]) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
    print("Missing Values:")
    print(missing_df[missing_df['Missing Values'] > 0])
    print("\n")

    print("Descriptive Statistics:")
    print(df.describe(include='all'))
    print("\n")

    unique_values = df.nunique()
    print("Unique Values:")
    print(unique_values)
    print("\n")

    numeric_data = df.select_dtypes(include=[np.number])
    if numeric_data.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        if save:
            plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'))
        plt.close()

    for column in numeric_data.columns:
        plt.figure()
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        if save:
            plt.savefig(os.path.join(save_dir, f'distribution_{column}.png'))
        plt.close()

    for column in df.select_dtypes(include=['object', 'category']).columns:
        plt.figure()
        sns.countplot(y=column, df=df)
        plt.title(f'Distribution of {column}')
        if save:
            plt.savefig(os.path.join(save_dir, f'countplot_{column}.png'))
        plt.close()

    print("EDA is complete.")
perform_eda(df,True)

# df.columns

df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)
df['total_score'] = df[['math_score', 'reading_score', 'writing_score']].sum(axis=1)
df['score_range'] = df[['math_score', 'reading_score', 'writing_score']].max(axis=1) - df[['math_score', 'reading_score', 'writing_score']].min(axis=1)
df['is_passing_math'] = df['math_score'].apply(lambda x: 1 if x >= 50 else 0)
df['is_passing_reading'] = df['reading_score'].apply(lambda x: 1 if x >= 50 else 0)
df['is_passing_writing'] = df['writing_score'].apply(lambda x: 1 if x >= 50 else 0)
df['overall_passing'] = df.apply(lambda row: 1 if row['is_passing_math'] == 1 and row['is_passing_reading'] == 1 and row['is_passing_writing'] == 1 else 0, axis=1)
df['gender_race_combo'] = df['gender'] + '_' + df['race_ethnicity']
df['education_lunch_combo'] = df['parental_level_of_education'] + '_' + df['lunch']
df['test_prep_effectiveness'] = df.apply(lambda row: 1 if row['test_preparation_course'] == 'completed' and row['total_score'] > df['total_score'].mean() else 0, axis=1)

def plot_histogram(data, x, bins, kde, color, title, hue=None, save=False):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].set_title(f'Distribution of {title}')
    axs[1].set_title(f'Distribution of {title} by Gender')
    sns.histplot(data=data, x=x, bins=bins, kde=kde, color=color, ax=axs[0])
    sns.histplot(data=data, x=x, kde=kde, hue=hue, ax=axs[1])
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'Dist_of_{title.replace(" ", "_")}_hist.png'))
    plt.tight_layout()
    plt.show()

def plot_histogram_by_category(data, x, category, title, hue_category, save=False):
    fig, axs = plt.subplots(1, 3, figsize=(25, 6))
    sns.histplot(data=data, x=x, kde=True, hue=hue_category, ax=axs[0])
    axs[0].set_title(f'Distribution of {title} by {category}')
    sns.histplot(data=data[data['gender'] == 'female'], x=x, kde=True, hue=hue_category, ax=axs[1])
    axs[1].set_title(f'Distribution of {title} by {category} (Female)')
    sns.histplot(data=data[data['gender'] == 'male'], x=x, kde=True, hue=hue_category, ax=axs[2])
    axs[2].set_title(f'Distribution of {title} by {category} (Male)')
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'Dist_of_{title.replace(" ", "_")}_by_{category.replace(" ", "_")}_hist.png'))
    plt.tight_layout()
    plt.show()

def plot_violin(data, columns, titles, colors, save=False):
    fig, axs = plt.subplots(1, len(columns), figsize=(25, 6))
    for i, col in enumerate(columns):
        sns.violinplot(y=col, data=data, color=colors[i], linewidth=3, ax=axs[i])
        axs[i].set_title(titles[i])
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'Dist_of_{"_".join([t.replace(" ", "_") for t in titles])}_violin.png'))
    plt.tight_layout()
    plt.show()

def plot_pie(data, column, labels, colors, title, ax, save=False):
    size = data[column].value_counts()
    ax.pie(size, colors=colors, labels=labels, autopct='%.2f%%')
    ax.set_title(title, fontsize=20)
    ax.axis('off')
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{title.replace(" ", "_")}_pie.png'))

def plot_count_pie(data, column, explode, title, pie_title, save=False):
    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.countplot(x=column, data=data, palette='bright', ax=ax[0], saturation=0.95)
    for container in ax[0].containers:
        ax[0].bar_label(container, color='black', size=20)
    ax[0].set_title(title, fontsize=20)

    plt.pie(x=data[column].value_counts(), labels=data[column].value_counts().index, explode=explode, autopct='%1.1f%%', shadow=True)
    plt.title(pie_title, fontsize=20)
    plt.tight_layout()
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{title.replace(" ", "_")}_count_pie.png'))
    plt.show(block=True)

def plot_bar(data, x, y, title, colors, save=False):
    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.countplot(x=x, data=data, palette=colors, ax=ax[0], saturation=0.95)
    for container in ax[0].containers:
        ax[0].bar_label(container, color='black', size=20)
    plt.pie(x=data[x].value_counts(), labels=y, explode=[0, 0.1], autopct='%1.1f%%', shadow=True, colors=colors)
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{title.replace(" ", "_")}_bar.png'))
    plt.tight_layout()
    plt.show()

def plot_grouped_bar(data, x_labels, group_labels, title, colors, save=False):
    gender_group = data.groupby('gender')[x_labels].mean()
    plt.figure(figsize=(10, 8))
    X_axis = np.arange(len(group_labels))
    for i, gender in enumerate(gender_group.index):
        plt.bar(X_axis + (i - 0.5) * 0.4, gender_group.loc[gender], 0.4, label=gender, color=colors[i])
    plt.xticks(X_axis, group_labels)
    plt.ylabel("Marks")
    plt.title(title, fontweight='bold')
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{title.replace(" ", "_")}_grouped_bar.png'))
    plt.legend()
    plt.show()

def plot_boxplots(df, columns, colors, figsize=(16, 5), save=False):
    num_plots = len(columns)
    plt.figure(figsize=figsize)

    for i, (col, color) in enumerate(zip(columns, colors)):
        plt.subplot(1, num_plots, i + 1)
        sns.boxplot(y=df[col], color=color)
        plt.title(col)
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, 'boxplot.png'))
    plt.tight_layout()
    plt.show()

# Use the functions
plot_histogram(df, 'average_score', bins=30, kde=True, color='g', title='Average Score', hue='gender', save=True)
plot_histogram(df, 'total_score', bins=30, kde=True, color='g', title='Total Score', hue='gender', save=True)
plot_histogram_by_category(df, 'average_score', 'Lunch', 'Average Score', 'lunch', save=True)
plot_histogram_by_category(df, 'average_score', 'Parental Level of Education', 'Average Score', 'parental_level_of_education', save=True)
plot_histogram_by_category(df, 'average_score', 'Race/Ethnicity', 'Average Score', 'race_ethnicity', save=True)
plot_violin(df, ['math_score', 'reading_score', 'writing_score'], ['MATH SCORES', 'READING SCORES', 'WRITING SCORES'], ['red', 'green', 'blue'], save=True)

fig, axs = plt.subplots(1, 5, figsize=(30, 12))
plot_pie(df, 'gender', ['Female', 'Male'], ['red', 'green'], 'Gender', axs[0], save=True)
plot_pie(df, 'race_ethnicity', df['race_ethnicity'].unique(), ['red', 'green', 'blue', 'cyan', 'orange'], 'Race/Ethnicity', axs[1], save=True)
plot_pie(df, 'lunch', ['Standard', 'Free/Reduced'], ['red', 'green'], 'Lunch', axs[2], save=True)
plot_pie(df, 'test_preparation_course', ['None', 'Completed'], ['red', 'green'], 'Test Course', axs[3], save=True)
plot_pie(df, 'parental_level_of_education', df['parental_level_of_education'].unique(), ['red', 'green', 'blue', 'cyan', 'orange', 'grey'], 'Parental Education', axs[4], save=True)
plt.tight_layout()
plt.show()

plot_bar(df, 'gender', ['Male', 'Female'], 'Gender Distribution', ['#ff4d4d', '#ff8000'])

plot_grouped_bar(df, ['average_score', 'math_score'], ['Total Average', 'Math Average'], 'Total average vs Math average marks of both the genders', ['blue', 'orange'],True)
plot_count_pie(df, 'race_ethnicity', explode=[0.1, 0, 0, 0, 0], title='Race/Ethnicity Distribution', pie_title='Race/Ethnicity Pie Chart')

plot_boxplots(df,['math_score','reading_score','writing_score','average_score'],['skyblue','hotpink','yellow','lightgreen'],save=True)

sns.pairplot(df,hue = 'gender')
plt.show(block=True)

df_ = df.copy()
# df_.to_csv('dataformodel.csv', index=False)




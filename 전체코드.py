import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Regression Libraries
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 파일 경로
train_file_path = r"C:\Users\mytoo\OneDrive\바탕 화면\소득예측\train.csv"
test_file_path = r"C:\Users\mytoo\OneDrive\바탕 화면\소득예측\test.csv"
sample_submission_file_path = r"C:\Users\mytoo\OneDrive\바탕 화면\소득예측\sample_submission.csv"

# 데이터 로드
train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)
sample_submission = pd.read_csv(sample_submission_file_path)

train.info()

# 타깃 밸류인 income 확인해보기
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
train['Income'].hist(bins=50, ax=axes[0])
axes[0].set_title('Histogram')
train['Income'].plot(kind='box', ax=axes[1])
axes[1].set_title('Boxplot')
plt.tight_layout()
plt.show()

# 이상치 제거
idx = train[(train['Income'] >= 5000) & (train['Income'] < 10000) & (train['Income_Status'] == 'Under Median')].index
train.drop(idx, axis=0, inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
train['Gains'].hist(bins=50, ax=axes[0])
axes[0].set_title('Histogram')
train['Gains'].plot(kind='box', ax=axes[1])
axes[1].set_title('Boxplot')
plt.tight_layout()
plt.show()

train['Gains'].head()
train.loc[(train['Gains'] == 99999), 'Gains'] = 0

def make_derived(df):
    df['Immigrant_Background'] = np.where((df['Birth_Country'] == 'US') & (df['Birth_Country (Father)'] == 'US'), 'Native', 'Immigrant')
    df['Total Income indicator'] = (df['Gains'] - df['Losses'] + df['Dividends'])
    df['Total Income indicator'] = df['Total Income indicator'] / np.sqrt(np.sum(df['Total Income indicator']**2))
    return df

make_derived(train)
make_derived(test)

# Income_Status가 'Unknown'인 행들의 Income 값이 어떤 분포를 가지고 있는지 이해하기 위함
print(train[(train['Income_Status'] == 'Unknown')]['Income'].describe())

# 'Income' 이상치 확인
plt.figure(figsize=(10, 6))
plt.boxplot(train['Income'])
plt.title('Box Plot of Income')
plt.ylabel('Income')
plt.show()

over_idx = train[train['Income'] > 7500].index
print(over_idx)
train.drop(over_idx, axis=0, inplace=True)

# test 파일의 결측값 확인
missing_test = test[test.isnull().any(axis=1)]
print(missing_test)

# test 파일의 결측값이 있는 행 삭제
test_cleaned = test.dropna()

# 결과 확인
print("결측값이 있는 행이 삭제된 test DataFrame:")
print(test_cleaned)

# 인덱스 리셋
train.reset_index(drop=True, inplace=True)
test_cleaned.reset_index(drop=True, inplace=True)

# 'ID' 열 제거
test_ids = test_cleaned['ID']
train = train.drop('ID', axis=1)
test_cleaned = test_cleaned.drop('ID', axis=1)

# 범주형 변수 확인
categorical_columns = train.select_dtypes(include=['object']).columns
print("\n범주형 컬럼:")
print(categorical_columns)

# 라벨 인코딩
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test_cleaned[col] = le.transform(test_cleaned[col].astype(str))
    label_encoders[col] = le

# 독립 변수(X)와 종속 변수(y) 분리
X = train.drop(['Income'], axis=1)
y = train['Income']

# train_test_split을 사용하여 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n훈련용 데이터셋 크기: ", X_train.shape)
print("검증용 데이터셋 크기: ", X_val.shape)

# 범주형 변수 인덱스 지정
cat_features = [X.columns.get_loc(col) for col in categorical_columns]

# CatBoostRegressor 모델 학습
model = CatBoostRegressor(random_state=42, cat_features=cat_features, verbose=500)
model.fit(X_train, y_train)

# 검증용 데이터셋으로 예측 수행
y_pred = model.predict(X_val)

# 예측 성능 평가
r2 = r2_score(y_val, y_pred)
print("\nR^2 Score: ", r2)

# 테스트 데이터 예측 수행
test_predictions = model.predict(test_cleaned)

# 결과를 데이터프레임으로 저장
submission = pd.DataFrame({'ID': test_ids, 'Income': test_predictions})

# 예측 결과를 CSV 파일로 저장
submission_file_path = r"C:\Users\mytoo\OneDrive\바탕 화면\소득예측\submission.csv"
submission.to_csv(submission_file_path, index=False)
print(submission_file_path)
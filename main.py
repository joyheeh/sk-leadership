import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import numpy as np
from collections import Counter
from openai import OpenAI
import base64
from collections import Counter
import re

import io
import openpyxl
import textwrap


# 사이드바에 OpenAI API 키 입력 필드 추가
st.sidebar.header("OpenAI API 설정")
api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
client = OpenAI(api_key=api_key)

# NLTK 데이터 다운로드
nltk.download('stopwords')
nltk.download('punkt')

# 데이터 불러오기 및 전처리 함수
@st.cache_data
def load_data():
    df = pd.read_excel("임원분석전처리.xlsx", index_col=0)
    # df = df.sample(10)
    return df

def process_scores(score_string):
    if pd.isna(score_string):
        return []
    return [int(score) for score in score_string.split(' // ') if score.isdigit()]

df = load_data()

# 데이터 전처리
df['연령대'] = pd.cut(df['만나이'], bins=[0, 40, 50, 60, 100], labels=['40세 미만', '40대', '50대', '60세 이상'])

score_columns = ['동료_NO9 의사소통 원활', '동료_NO10 긴밀한 협조 가능', '동료_NO11 전사 관점의 성과극대화 노력', '부하_전년대비 조직변화-정량', '부하_리더의 경험수준']
for col in score_columns:
    df[f'{col}_processed'] = df[col].apply(process_scores)
    df[f'{col}_mean'] = df[f'{col}_processed'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)

# 협업 수준 계산
def calculate_collaboration_score(row):
    collaboration_columns = ['동료_NO9 의사소통 원활_mean', '동료_NO10 긴밀한 협조 가능_mean', '동료_NO11 전사 관점의 성과극대화 노력_mean']
    scores = [row[col] for col in collaboration_columns if pd.notnull(row[col])]
    return sum(scores) / len(scores) if scores else None

df['collaboration_score'] = df.apply(calculate_collaboration_score, axis=1)

# 텍스트 전처리 함수
def preprocess_text(text):
    if pd.isna(text):
        return ""
    return text  # 한국어 전처리는 LLM에 맡김

# LLM을 사용한 텍스트 분석 함수
def analyze_text_with_llm(texts, task):
    prompt = f"{task}:\n\n" + "\n".join(texts)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        n=1,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()
# LLM을 사용한 키워드 추출
def extract_keywords_with_llm(col_name, text, n=5):
    if pd.isna(text):
        return ["정보 없음"]
    prompt = f"다음 텍스트에서 {col_name}를 설명하는 가장 중요한 키워드 {n}개를 추출해주세요. 키워드만 쉼표로 구분하여 나열해주세요:\n\n{text}"
    response = analyze_text_with_llm([prompt], "키워드 추출")
    keywords = [keyword.strip() for keyword in response.split(',')]
    return keywords[:n]

# 키워드 추출 및 빈도 계산 함수
def extract_and_count_keywords(text, keywords):
    word_counts = Counter(re.findall(r'\w+', text.lower()))
    return {keyword: word_counts[keyword.lower()] for keyword in keywords}

# 임원 간 비교 함수
def compare_executives(exec1, exec2):
    exec1 = str(exec1)
    exec2 = str(exec2)
    exec1_data = df[df['식별'] == int(exec1)].iloc[0]
    exec2_data = df[df['식별'] == int(exec2)].iloc[0]

    comparison_data = {
        '동료임원이 인식한 협업 수준': [exec1_data['collaboration_score'], exec2_data['collaboration_score']],
        '부하 구성원이 평가한 리더의 경험 수준': [exec1_data['부하_리더의 경험수준_mean'], exec2_data['부하_리더의 경험수준_mean']],
        '임원 본인의 조직 변화 인식': [exec1_data['전년대비 조직변화수준 - 정성'], exec2_data['전년대비 조직변화수준 - 정성']]
    }
        # 평가 점수 레이더 차트
    st.subheader("평가 점수 비교")
    categories = [col.split('_')[1] for col in score_columns]
    values_exec1 = [exec1_data[col + '_mean'] for col in score_columns]
    values_exec2 = [exec2_data[col + '_mean'] for col in score_columns]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_exec1,
        theta=categories,
        fill='toself',
        name=f"기존 선택한 임원 ({exec1})"  # Updated legend label
    ))

    fig.add_trace(go.Scatterpolar(
        r=values_exec2,
        theta=categories,
        fill='toself',
        name=f"비교할 임원 ({exec2})"  # Updated legend label
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=True
    )

    st.plotly_chart(fig)



    # 텍스트 데이터 비교
    st.subheader("텍스트 데이터 비교")

    exclude_columns = ['식별', '생년', '구분', '부하_구분', '상사_구분', '동료_구분']
    text_columns = [col for col in df.select_dtypes(include=['object']).columns if col not in exclude_columns]

    # 사용자가 비교하고 싶은 필드 선택
    selected_fields = st.multiselect("비교할 필드를 선택하세요", text_columns)

    for field in selected_fields:
        st.subheader(f"{field} 비교")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"{exec1}:")
            st.write(exec1_data[field])
        with col2:
            st.write(f"{exec2}:")
            st.write(exec2_data[field])



# 협업 네트워크 그래프 생성 함수
def create_collaboration_network(df, threshold):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['식별'], 자회사=row['자회사'], 부서=row['부서'])

    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i < j:  # 중복 방지
                score = min(row1['collaboration_score'], row2['collaboration_score'])
                if score > threshold:
                    G.add_edge(row1['식별'], row2['식별'], weight=score)

    # 그래프 그리기
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='협업 점수',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_info = df[df['식별'] == adjacencies[0]].iloc[0]
        node_text.append(f"임원: {adjacencies[0]}<br>자회사: {node_info['자회사']}<br>부서: {node_info['부서']}<br>협업 점수: {node_info['collaboration_score']:.2f}")

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='임원 간 협업 네트워크',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="협업 점수가 높을수록 연결이 많습니다",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

# 페이지 선택을 위한 사이드바 설정
st.sidebar.title("SK 그룹 임원 데이터 분석")
page = st.sidebar.radio("페이지 선택", ["전체 분석 개요", "상세 분석"])

if page == "전체 분석 개요":
    st.title("SK 그룹 임원 데이터 분석 - 전체 개요")

    # 데이터 필터링 옵션
    st.sidebar.header("데이터 필터링")
    filter_option = st.sidebar.radio("필터링 옵션", ["전체", "특성별 선택"])
    
    if filter_option == "특성별 선택":
        # 각 특성에 대해 "전체" 옵션 추가
        company_options = ["전체"] + list(df['자회사'].unique())
        department_options = ["전체"] + list(df['부서'].unique())
        gender_options = ["전체"] + list(df['성별'].unique())
        age_group_options = ["전체"] + list(df['연령대'].unique())

        selected_company = st.sidebar.multiselect("자회사 선택", company_options, default=["전체"])
        selected_department = st.sidebar.multiselect("부서 선택", department_options, default=["전체"])
        selected_gender = st.sidebar.multiselect("성별 선택", gender_options, default=["전체"])
        selected_age_group = st.sidebar.multiselect("연령대 선택", age_group_options, default=["전체"])
        
        # 필터링 적용
        filtered_df = df.copy()
        if "전체" not in selected_company:
            filtered_df = filtered_df[filtered_df['자회사'].isin(selected_company)]
        if "전체" not in selected_department:
            filtered_df = filtered_df[filtered_df['부서'].isin(selected_department)]
        if "전체" not in selected_gender:
            filtered_df = filtered_df[filtered_df['성별'].isin(selected_gender)]
        if "전체" not in selected_age_group:
            filtered_df = filtered_df[filtered_df['연령대'].isin(selected_age_group)]
    else:
        filtered_df = df

    
    # 1. 데이터 Overview
    st.header("1. 데이터 Overview")
    
    st.subheader("데이터 정보")
    # 각 변수별 유효한 데이터 갯수
    valid_counts = filtered_df.notna().sum()
    valid_counts = pd.DataFrame(valid_counts).T  # 행열 전환
    valid_counts.index = ['유효한 데이터 갯수']
    st.write(valid_counts)

    # 자회사 및 부서별 임원 분포
    st.subheader("자회사 및 부서별 임원 분포")
    
    # 선택 옵션
    distribution_option = st.radio("분포 기준 선택", ["자회사", "부서"])
    
    if distribution_option == "자회사":
        group_col = '자회사'
    else:
        group_col = '부서'
    
    # 데이터 집계
    counts = filtered_df.groupby(group_col).agg({
        '식별': 'count',
        '성별': lambda x: (x == '남성').mean(),
        '진단사유': lambda x: x.value_counts(normalize=True).to_dict()
    }).reset_index()
    counts.columns = ['구분', '임원 수', '남성 비율', '진단사유 비율']
    
    # 호버 텍스트 생성 함수
    def create_hover_text(row):
        hover_text = f"임원 수: {row['임원 수']}<br>"
        hover_text += f"성별 비율: 남성 {row['남성 비율']:.0%}, 여성 {1-row['남성 비율']:.0%}<br>"
        hover_text += "진단사유 비율:<br>"
        for reason, ratio in row['진단사유 비율'].items():
            hover_text += f"  {reason}: {ratio:.0%}<br>"
        return hover_text
    
    counts['hover_text'] = counts.apply(create_hover_text, axis=1)
    
    # 그래프 생성
    fig = px.bar(counts, x='구분', y='임원 수')
    fig.update_layout(
        xaxis_title=distribution_option,
        yaxis_title="임원 수",
        hovermode="x unified"
    )
    
    # 호버 텍스트 설정
    fig.update_traces(
        hovertemplate="%{customdata}",
        customdata=counts['hover_text']
    )
    
    st.plotly_chart(fig)

    # 성별 분포
    st.subheader("성별 분포")
    gender_counts = filtered_df['성별'].value_counts().reset_index()
    gender_counts.columns = ['성별', '임원 수']
    fig = px.pie(gender_counts, names='성별', values='임원 수', title='성별 분포',
                 hover_data=['임원 수'], labels={'임원 수': '임원 수'})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)

    # 연령대 분포
    st.subheader("연령대 분포")
    age_counts = filtered_df['만나이'].value_counts().reset_index()
    age_counts.columns = ['만나이', '임원 수']
    fig = px.bar(age_counts, x='만나이', y='임원 수', title='연령대 분포')
    st.plotly_chart(fig)

    # 평균 점수 분포
    st.subheader("평균 점수 분포")
    score_data = pd.melt(filtered_df[[col + '_mean' for col in score_columns]], var_name='Category', value_name='Score')
    fig = px.box(score_data, x='Category', y='Score', points="all")
    st.plotly_chart(fig)


    # 추가 인사이트 및 시각화
    st.subheader("동료임원이 평가한 협업 수준 추가 인사이트")

    # 연령대별 협업 수준 분포
    st.subheader("연령대별 협업 수준 분포")
    fig = px.box(filtered_df, x='연령대', y='collaboration_score', points="all")
    st.plotly_chart(fig)

    # 자회사별 평균 협업 수준
    st.subheader("자회사별 평균 협업 수준")
    company_collaboration = filtered_df.groupby('자회사')['collaboration_score'].mean().sort_values(ascending=False)
    fig = px.bar(company_collaboration, x=company_collaboration.index, y='collaboration_score')
    st.plotly_chart(fig)

    # 성별에 따른 협업 수준 차이
    st.subheader("성별에 따른 협업 수준 차이")
    fig = px.box(filtered_df, x='성별', y='collaboration_score', points="all")
    st.plotly_chart(fig)

    # 진단사유에 따른 협업 수준 차이
    st.subheader("진단사유에 따른 협업 수준 차이")
    fig = px.box(filtered_df, x='진단사유', y='collaboration_score', points="all")
    st.plotly_chart(fig)

    # # 협업 네트워크 그래프
    # if st.checkbox("협업 네트워크 그래프 표시"):
    #     threshold = st.slider("협업 연결 임계값", min_value=0.0, max_value=5.0, value=3.5, step=0.1)
    #     fig = create_collaboration_network(df, threshold)
    #     st.plotly_chart(fig)

    # 데이터 다운로드
    st.header("분석 데이터 다운로드")
    if st.button("데이터 다운로드"):
        # 엑셀 파일 생성
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        # 파일 다운로드 링크 생성
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="sk_executives_data.xlsx">Excel 파일 다운로드</a>'
        st.markdown(href, unsafe_allow_html=True)

    # 피드백 섹션
    st.header("피드백")
    feedback = st.text_area("이 분석 도구에 대한 의견이나 개선사항을 알려주세요:")
    if st.button("피드백 제출"):
        st.success("피드백이 제출되었습니다. 감사합니다!")

elif page == "상세 분석":
    st.title("SK 그룹 임원 데이터 분석 - 상세 분석")

    # 분석 대상 선택
    analysis_target = st.sidebar.radio("분석 대상 선택", ["개별 임원", "그룹별 분석"])

    if analysis_target == "개별 임원":
        executives = df['식별'].unique()
        selected_executive = st.sidebar.selectbox("분석할 임원 선택", executives)
        exec_data = df[df['식별'] == selected_executive].iloc[0]
    else:
        feature = st.sidebar.selectbox("그룹화할 특성 선택", ['자회사', '부서', '성별', '진단사유'])
        groups = df[feature].unique()
        selected_group = st.sidebar.selectbox(f"분석할 {feature} 선택", groups)
        group_df = df[df[feature] == selected_group]

    # 분석 항목 선택
    analysis_options = st.sidebar.multiselect(
        "분석 항목 선택",
        ["2. 특정 임원 상세 정보", "3. SK 그룹 임원 Trend 분석", 
         "4. SK 그룹 임원 강/약점 인식 차이 분석", "5. 임원과 부하 구성원 인식 차이 분석", 
         "6. 협업 수준에 따른 임원 그룹 간 차이점 분석", "7. 종합 분석 및 제언"]
    )

    # 선택된 분석 항목에 따라 분석 실행
    if "2. 특정 임원 상세 정보" in analysis_options and analysis_target == "개별 임원":
        st.header("2. 특정 임원 상세 정보")
        
        st.subheader(f"{exec_data['자회사']} - {exec_data['부서']} 임원 정보")
        st.write(f"나이: {exec_data['만나이']}세")
        st.write(f"성별: {exec_data['성별']}")
        st.write(f"진단사유: {exec_data['진단사유']}")

        # 경력 및 비전 시각화
        st.subheader("경력 터닝포인트 및 커리어 비전")

        # 텍스트 줄바꿈 함수
        def wrap_text(text, width=50):
            if pd.isna(text):
                return "정보 없음"
            return "\n".join(textwrap.wrap(str(text), width=width))

        # 경력 터닝포인트와 커리어 비전 데이터 가져오기
        turning_point = wrap_text(exec_data['경력상 Truning point'])
        career_vision = wrap_text(exec_data['career vision'])

        # 두 열로 나누어 표시
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("경력 터닝포인트")
            st.text(turning_point)

        with col2:
            st.subheader("커리어 비전")
            st.text(career_vision)

        st.subheader("주요 키워드")

        # 키워드 추출
        turning_point_keywords = extract_keywords_with_llm('경력상 Truning point', exec_data['경력상 Truning point'])
        career_vision_keywords = extract_keywords_with_llm('career vision', exec_data['career vision'])

        # 키워드 표시
        st.write("경력 터닝포인트 주요 키워드:")
        st.write(", ".join(turning_point_keywords))

        st.write("\n커리어 비전 주요 키워드:")
        st.write(", ".join(career_vision_keywords))



        # 평가 점수 레이더 차트
        st.subheader("객관식 평가 점수")
        categories = [col.split('_')[1] for col in score_columns]
        values = [exec_data[col + '_mean'] for col in score_columns]

        # 전체 데이터의 평균 계산
        avg_values = [df[col + '_mean'].mean() for col in score_columns]

        fig = go.Figure()

        # 전체 평균 데이터 추가
        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=categories,
            fill='toself',
            name='전체 평균',
            line=dict(color='rgba(255, 0, 0, 0.5)')  # 빨간색, 반투명
        ))

        # 선택된 임원의 데이터 추가
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='선택된 임원',
            line=dict(color='rgba(0, 0, 255, 0.8)')  # 파란색
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True
        )
        st.plotly_chart(fig)

        # 임원 간 비교 기능 추가
        st.header("임원 간 비교")
        if st.checkbox("임원 간 비교 기능 활성화"):
            exec1 = selected_executive  # 2번 항목에서 선택된 임원
            other_executives = [exec for exec in df['식별'].unique() if exec != exec1]
            exec2 = st.selectbox("비교할 임원 선택", other_executives, key='exec2')

            if exec2:
                compare_executives(exec1, exec2)

    if "3. SK 그룹 임원 Trend 분석" in analysis_options:
        st.header("3. SK 그룹 임원 Trend 분석")
        
        text_columns = ['경력상 Truning point', 'career vision', '중요한 전략적 Agenda 및 해결해야 할 과제', 
                        '조직특성', '강점', '개발필요점', '본인 리더십 개발을 위한 노력', 
                        '전년대비 조직변화수준 - 정성', '전년대비 조직변화수준 - 서술']
        selected_text_column = st.selectbox("분석할 서술형 항목 선택", text_columns)
        
        if analysis_target == "개별 임원":
            texts = [exec_data[selected_text_column]]
        else:
            texts = group_df[selected_text_column].dropna().tolist()
        



    if "4. SK 그룹 임원 강/약점 인식 차이 분석" in analysis_options:
        st.header("4. SK 그룹 임원 강/약점 인식 차이 분석")
        
        if analysis_target == "개별 임원":
            strength_data = {
                '본인': exec_data['강점'],
                '부하': exec_data['부하_강점'],
                '상사': exec_data['상사_강점을 더욱 강하게 하는 솔선수범 및 진정성 있는 소통리더십'],
                '동료': exec_data['동료_강점']
            }
            weakness_data = {
                '본인': exec_data['개발필요점'],
                '부하': exec_data['부하_개선필요점'],
                '상사': exec_data['상사_개발 필요점'],
                '동료': exec_data['동료_개선할점']
            }
        else:
            strength_data = {
                '본인': group_df['강점'].tolist(),
                '부하': group_df['부하_강점'].tolist(),
                '상사': group_df['상사_강점을 더욱 강하게 하는 솔선수범 및 진정성 있는 소통리더십'].tolist(),
                '동료': group_df['동료_강점'].tolist()
            }
            weakness_data = {
                '본인': group_df['개발필요점'].tolist(),
                '부하': group_df['부하_개선필요점'].tolist(),
                '상사': group_df['상사_개발 필요점'].tolist(),
                '동료': group_df['동료_개선할점'].tolist()
            }

        # 강점 분석
        st.subheader("강점 분석")
        
        # 워드클라우드 생성
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        for idx, (perspective, text) in enumerate(strength_data.items()):
            if isinstance(text, list):
                text = ' '.join(text)
            wordcloud = WordCloud(width=400, height=400, background_color='white').generate(text)
            axs[idx//2, idx%2].imshow(wordcloud, interpolation='bilinear')
            axs[idx//2, idx%2].set_title(perspective)
            axs[idx//2, idx%2].axis('off')
        st.pyplot(fig)

        # 약점 분석
        st.subheader("약점 분석")
        
        # 워드클라우드 생성
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        for idx, (perspective, text) in enumerate(weakness_data.items()):
            if isinstance(text, list):
                text = ' '.join(text)
            wordcloud = WordCloud(width=400, height=400, background_color='white').generate(text)
            axs[idx//2, idx%2].imshow(wordcloud, interpolation='bilinear')
            axs[idx//2, idx%2].set_title(perspective)
            axs[idx//2, idx%2].axis('off')
        st.pyplot(fig)

    if "5. 임원과 부하 구성원 인식 차이 분석" in analysis_options:
        st.header("5. 임원과 부하 구성원 인식 차이 분석")
        
        if analysis_target == "개별 임원":
            exec_perception = exec_data['전년대비 조직변화수준 - 정성']
            subordinate_perception = exec_data['부하_전년대비 조직변화-정량_mean']
            
            st.subheader("조직 변화에 대한 인식 차이")
            fig = go.Figure(data=[
                go.Bar(name='임원', x=['조직 변화 인식'], y=[exec_perception]),
                go.Bar(name='부하', x=['조직 변화 인식'], y=[subordinate_perception])
            ])
            fig.update_layout(barmode='group')
            st.plotly_chart(fig)
            
            st.subheader("경험 수준에 대한 인식 차이")
            exec_experience = exec_data['나의 경험 수준']
            subordinate_experience = exec_data['부하_리더의 경험수준_mean']
            
            fig = go.Figure(data=[
                go.Bar(name='임원', x=['경험 수준'], y=[exec_experience]),
                go.Bar(name='부하', x=['경험 수준'], y=[subordinate_experience])
            ])
            fig.update_layout(barmode='group')
            st.plotly_chart(fig)
        else:
            exec_perception = group_df['전년대비 조직변화수준 - 정성'].mean()
            subordinate_perception = group_df['부하_전년대비 조직변화-정량_mean'].mean()
            
            st.subheader("조직 변화에 대한 인식 차이")
            fig = go.Figure(data=[
                go.Bar(name='임원', x=['조직 변화 인식'], y=[exec_perception]),
                go.Bar(name='부하', x=['조직 변화 인식'], y=[subordinate_perception])
            ])
            fig.update_layout(barmode='group')
            st.plotly_chart(fig)
            
            st.subheader("경험 수준에 대한 인식 차이")
            exec_experience = group_df['나의 경험 수준'].mean()
            subordinate_experience = group_df['부하_리더의 경험수준_mean'].mean()
            
            fig = go.Figure(data=[
                go.Bar(name='임원', x=['경험 수준'], y=[exec_experience]),
                go.Bar(name='부하', x=['경험 수준'], y=[subordinate_experience])
            ])
            fig.update_layout(barmode='group')
            st.plotly_chart(fig)

    if "6. 협업 수준에 따른 임원 그룹 간 차이점 분석" in analysis_options:
        st.header("6. 협업 수준에 따른 임원 그룹 간 차이점 분석")
        
        median_score = df['collaboration_score'].median()
        high_collaboration_group = df[df['collaboration_score'] > median_score]
        low_collaboration_group = df[df['collaboration_score'] <= median_score]

        st.write(f"전체 임원 수: {len(df)}")
        st.write(f"협업 수준 높은 그룹 임원 수: {len(high_collaboration_group)}")
        st.write(f"협업 수준 낮은 그룹 임원 수: {len(low_collaboration_group)}")

        # 협업 수준에 따른 리더십 특성 비교
        st.subheader("협업 수준에 따른 리더십 특성 비교")
        
        leadership_characteristics = ['부하_리더십이미지', '부하_강점', '상사_강점을 더욱 강하게 하는 솔선수범 및 진정성 있는 소통리더십']
        
        for characteristic in leadership_characteristics:
            high_group_text = ' '.join(high_collaboration_group[characteristic].dropna())
            low_group_text = ' '.join(low_collaboration_group[characteristic].dropna())
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            wordcloud_high = WordCloud(width=400, height=400, background_color='white').generate(high_group_text)
            ax1.imshow(wordcloud_high, interpolation='bilinear')
            ax1.set_title('협업 수준 높은 그룹')
            ax1.axis('off')
            
            wordcloud_low = WordCloud(width=400, height=400, background_color='white').generate(low_group_text)
            ax2.imshow(wordcloud_low, interpolation='bilinear')
            ax2.set_title('협업 수준 낮은 그룹')
            ax2.axis('off')
            
            st.pyplot(fig)

        # 협업 수준에 따른 조직 변화 인식 비교
        st.subheader("협업 수준에 따른 조직 변화 인식 비교")
        
        high_group_change = high_collaboration_group['부하_전년대비 조직변화-정량_mean'].mean()
        low_group_change = low_collaboration_group['부하_전년대비 조직변화-정량_mean'].mean()
        
        fig = go.Figure(data=[
            go.Bar(name='협업 수준 높은 그룹', x=['조직 변화 인식'], y=[high_group_change]),
            go.Bar(name='협업 수준 낮은 그룹', x=['조직 변화 인식'], y=[low_group_change])
        ])
        fig.update_layout(barmode='group')
        st.plotly_chart(fig)

        # 협업 수준과 다른 요인들의 상관관계 분석
        st.subheader("협업 수준과 다른 요인들의 상관관계")
        
        correlation_factors = ['만나이', '부하_리더의 경험수준_mean', '부하_전년대비 조직변화-정량_mean']
        correlation_data = df[['collaboration_score'] + correlation_factors].corr()['collaboration_score'].drop('collaboration_score')
        
        fig = px.bar(x=correlation_data.index, y=correlation_data.values,
                     labels={'x': '요인', 'y': '협업 수준과의 상관계수'},
                     title='협업 수준과 다른 요인들의 상관관계')
        st.plotly_chart(fig)

    if "7. 종합 분석 및 제언" in analysis_options:
        st.header("종합 분석 및 제언")
        if analysis_target == "개별 임원":
            analysis_prompt = f"""
            다음은 {exec_data['자회사']} {exec_data['부서']} 임원의 분석 결과입니다:

            1. 강점: {exec_data['강점']}
            2. 개발필요점: {exec_data['개발필요점']}
            3. 부하 인식 리더십 이미지: {exec_data['부하_리더십이미지']}
            4. 상사 인식 강점: {exec_data['상사_강점을 더욱 강하게 하는 솔선수범 및 진정성 있는 소통리더십']}
            5. 협업 수준 점수: {exec_data['collaboration_score']}

            위 정보를 바탕으로 다음 사항들에 대해 분석해주세요:
            1. 이 임원의 주요 강점과 개선이 필요한 영역
            2. 리더십 효과성을 높이기 위한 구체적인 제안 3가지
            3. 협업 능력 향상을 위한 조언
            4. 조직 성과 향상을 위해 이 임원이 집중해야 할 영역
            """
        else:
            analysis_prompt = f"""
            다음은 {feature}가 {selected_group}인 임원 그룹의 분석 결과입니다:

            1. 그룹 내 임원 수: {len(group_df)}
            2. 평균 협업 수준 점수: {group_df['collaboration_score'].mean()}
            3. 주요 강점 키워드: {', '.join(group_df['강점'].str.cat().split()[:10])}
            4. 주요 개발필요점 키워드: {', '.join(group_df['개발필요점'].str.cat().split()[:10])}

            위 정보를 바탕으로 다음 사항들에 대해 분석해주세요:
            1. 이 그룹 임원들의 주요 강점과 개선이 필요한 공통적인 영역
            2. 그룹 전체의 리더십 효과성을 높이기 위한 구체적인 제안 3가지
            3. 그룹의 협업 능력 향상을 위한 조언
            4. 조직 성과 향상을 위해 이 그룹의 임원들이 집중해야 할 영역
            """

        analysis = analyze_text_with_llm([analysis_prompt], "아래의 지시사항에 따라 분석을 수행해주세요.")
        st.write(analysis)

# 임원 간 비교 함수
def compare_executives(exec1, exec2):
    exec1_data = df[df['식별'] == exec1].iloc[0]
    exec2_data = df[df['식별'] == exec2].iloc[0]

    # Convert exec1 and exec2 to strings explicitly
    exec1_name = str(exec1)
    exec2_name = str(exec2)

    comparison_data = {
        '동료임원이 인식한 협업 수준': [exec1_data['collaboration_score'], exec2_data['collaboration_score']],
        '부하 구성원이 평가한 리더의 경험 수준': [exec1_data['부하_리더의 경험수준_mean'], exec2_data['부하_리더의 경험수준_mean']],
        '임원 본인의 조직 변화 인식': [exec1_data['전년대비 조직변화수준 - 정성'], exec2_data['전년대비 조직변화수준 - 정성']]
    }

    # 평가 점수 레이더 차트
    st.subheader("객관식 평가 점수 비교")
    categories = [col.split('_')[1] for col in score_columns]
    values_exec1 = [exec1_data[col + '_mean'] for col in score_columns]
    values_exec2 = [exec2_data[col + '_mean'] for col in score_columns]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_exec1,
        theta=categories,
        fill='toself',
        name=exec1_name  # Use the string version
    ))

    fig.add_trace(go.Scatterpolar(
        r=values_exec2,
        theta=categories,
        fill='toself',
        name=exec2_name  # Use the string version
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=True
    )

    st.plotly_chart(fig)

    # 텍스트 데이터 비교
    st.subheader("텍스트 데이터 비교")

    exclude_columns = ['식별', '생년', '구분', '부하_구분', '상사_구분', '동료_구분']
    text_columns = [col for col in df.select_dtypes(include=['object']).columns if col not in exclude_columns]

    # 사용자가 비교하고 싶은 필드 선택
    selected_fields = st.multiselect("비교할 필드를 선택하세요", text_columns)

    for field in selected_fields:
        st.subheader(f"{field} 비교")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"{exec1_name}:")
            st.write(exec1_data[field])
        with col2:
            st.write(f"{exec2_name}:")
            st.write(exec2_data[field])

# 협업 네트워크 그래프 생성 함수
def create_collaboration_network(df, threshold):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['식별'], 자회사=row['자회사'], 부서=row['부서'])

    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i < j:  # 중복 방지
                score = min(row1['collaboration_score'], row2['collaboration_score'])
                if score > threshold:
                    G.add_edge(row1['식별'], row2['식별'], weight=score)

    # 그래프 그리기
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='협업 점수',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_info = df[df['식별'] == adjacencies[0]].iloc[0]
        node_text.append(f"임원: {adjacencies[0]}<br>자회사: {node_info['자회사']}<br>부서: {node_info['부서']}<br>협업 점수: {node_info['collaboration_score']:.2f}")

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='임원 간 협업 네트워크',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="협업 점수가 높을수록 연결이 많습니다",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig


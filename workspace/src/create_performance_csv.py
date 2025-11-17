#!/usr/bin/env python3
"""
Enhanced Features 최종 모델 성능을 CSV 파일로 저장
"""

import json
import pandas as pd

def create_performance_csv():
    """JSON 결과를 CSV로 변환"""
    
    # JSON 파일 로드
    with open('workspace/final_model_performance_analysis.json', 'r') as f:
        data = json.load(f)
    
    detailed_results = data['detailed_results']
    
    # CSV용 데이터 준비
    csv_data = []
    
    for dataset_name, result in detailed_results.items():
        csv_data.append({
            'Dataset': dataset_name,
            'Category': result['category'],
            'Task_Type': result['task_type'],
            'Metric_Type': result['metric_name'],
            'Performance': round(result['test_metric'], 4),
            'Model_Size_MB': round(result['file_size_mb'], 1),
            'Total_Samples': result['total_samples']
        })
    
    # DataFrame 생성 및 정렬
    df = pd.DataFrame(csv_data)
    
    # 카테고리별, 데이터셋명별로 정렬
    df = df.sort_values(['Category', 'Dataset'])
    
    # CSV 파일 저장
    output_file = 'workspace/enhanced_features_performance.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Enhanced Features 성능 결과 저장: {output_file}")
    print(f"📊 총 {len(df)}개 데이터셋 결과")
    
    # 요약 통계 출력
    print("\n📈 Summary Statistics:")
    
    # Classification 통계
    classification_df = df[df['Task_Type'] == 'classification']
    if len(classification_df) > 0:
        avg_auroc = classification_df['Performance'].mean()
        print(f"  Classification (AUROC): {avg_auroc:.4f} (avg) | {len(classification_df)} datasets")
    
    # Regression 통계  
    regression_df = df[df['Task_Type'] == 'regression']
    if len(regression_df) > 0:
        avg_mae = regression_df['Performance'].mean()
        print(f"  Regression (MAE): {avg_mae:.4f} (avg) | {len(regression_df)} datasets")
    
    # 카테고리별 통계
    print(f"\n📂 Category Breakdown:")
    category_summary = df.groupby('Category').agg({
        'Dataset': 'count',
        'Performance': 'mean',
        'Model_Size_MB': 'sum'
    }).round(4)
    category_summary.columns = ['Count', 'Avg_Performance', 'Total_Size_MB']
    print(category_summary.to_string())
    
    # CSV 파일 미리보기
    print(f"\n📄 CSV File Preview:")
    print(df.head(10).to_string(index=False))
    
    return output_file, df

def main():
    output_file, df = create_performance_csv()
    return output_file, df

if __name__ == "__main__":
    main()
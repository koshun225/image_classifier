"""
CRUD操作のユーティリティ関数
"""
from typing import List, Optional, Dict
from django.db.models import Count, Q
from .models import Theme, Label, TrainData, Model, ModelTrainData
import random


def get_theme(theme_id: int) -> Optional[Theme]:
    """テーマを取得"""
    try:
        return Theme.objects.get(id=theme_id)
    except Theme.DoesNotExist:
        return None


def get_all_themes() -> List[Theme]:
    """すべてのテーマを取得"""
    return Theme.objects.annotate(
        label_count=Count('labels', distinct=True),
        image_count=Count('traindata', distinct=True)
    ).all()


def create_theme(name: str, description: str = "") -> Theme:
    """テーマを作成"""
    return Theme.objects.create(name=name, description=description)


def update_theme(theme_id: int, name: str = None, description: str = None) -> Optional[Theme]:
    """テーマを更新"""
    theme = get_theme(theme_id)
    if theme is None:
        return None
    
    if name is not None:
        theme.name = name
    if description is not None:
        theme.description = description
    theme.save()
    return theme


def delete_theme(theme_id: int) -> bool:
    """テーマを削除"""
    theme = get_theme(theme_id)
    if theme is None:
        return False
    theme.delete()
    return True


def get_labels_by_theme(theme_id: int) -> List[Label]:
    """テーマのラベル一覧を取得"""
    return Label.objects.filter(theme_id=theme_id).order_by('id')


def get_label(theme_id: int, label_id: int) -> Optional[Label]:
    """ラベルを取得"""
    try:
        return Label.objects.get(id=label_id, theme_id=theme_id)
    except Label.DoesNotExist:
        return None


def create_label(theme_id: int, label_name: str) -> Optional[Label]:
    """ラベルを作成"""
    theme = get_theme(theme_id)
    if theme is None:
        return None
    
    try:
        return Label.objects.create(theme=theme, label_name=label_name)
    except Exception:
        return None


def update_label(theme_id: int, label_id: int, label_name: str) -> Optional[Label]:
    """ラベルを更新"""
    label = get_label(theme_id, label_id)
    if label is None:
        return None
    
    label.label_name = label_name
    label.save()
    return label


def delete_label(theme_id: int, label_id: int) -> bool:
    """ラベルを削除"""
    label = get_label(theme_id, label_id)
    if label is None:
        return False
    label.delete()
    return True


def get_traindata_by_theme(theme_id: int, split: str = None, label_id: int = None) -> List[TrainData]:
    """テーマの学習データを取得"""
    queryset = TrainData.objects.filter(theme_id=theme_id)
    
    if split:
        queryset = queryset.filter(split=split)
    
    if label_id:
        queryset = queryset.filter(label_id=label_id)
    
    return list(queryset.order_by('-created_at'))


def get_traindata(theme_id: int, traindata_id: int) -> Optional[TrainData]:
    """学習データを取得"""
    try:
        return TrainData.objects.get(id=traindata_id, theme_id=theme_id)
    except TrainData.DoesNotExist:
        return None


def create_traindata(theme_id: int, image, label_id: int = None, labeled_by: str = None) -> Optional[TrainData]:
    """学習データを作成"""
    theme = get_theme(theme_id)
    if theme is None:
        return None
    
    label = None
    if label_id:
        label = get_label(theme_id, label_id)
    
    return TrainData.objects.create(
        theme=theme,
        label=label,
        image=image,
        labeled_by=labeled_by
    )


def update_traindata_label(theme_id: int, traindata_id: int, label_id: int = None) -> Optional[TrainData]:
    """学習データのラベルを更新"""
    traindata = get_traindata(theme_id, traindata_id)
    if traindata is None:
        return None
    
    if label_id:
        label = get_label(theme_id, label_id)
        if label is None:
            return None
        traindata.label = label
    else:
        traindata.label = None
    
    traindata.save()
    return traindata


def delete_traindata(theme_id: int, traindata_id: int) -> bool:
    """学習データを削除"""
    traindata = get_traindata(theme_id, traindata_id)
    if traindata is None:
        return False
    traindata.delete()
    return True


def get_class_names_by_theme(theme_id: int) -> List[str]:
    """テーマのクラス名リストを取得"""
    labels = get_labels_by_theme(theme_id)
    return [label.label_name for label in labels]


def get_split_statistics(theme_id: int) -> Dict[str, int]:
    """データ分割統計を取得"""
    result = {
        'train': TrainData.objects.filter(theme_id=theme_id, split='train').count(),
        'valid': TrainData.objects.filter(theme_id=theme_id, split='valid').count(),
        'test': TrainData.objects.filter(theme_id=theme_id, split='test').count(),
        'unsplit': TrainData.objects.filter(theme_id=theme_id, split__isnull=True).count()
    }
    
    return result


def assign_splits_to_new_data(
    theme_id: int,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, int]:
    """
    未分割のデータをtrain/valid/testに分割（層化分割）
    
    各クラスの割合がtrain/valid/testで維持されるように分割します。
    
    Args:
        theme_id: テーマID
        train_ratio: 学習データ比率
        valid_ratio: 検証データ比率
        test_ratio: テストデータ比率
        random_seed: ランダムシード
    
    Returns:
        分割結果の統計
    """
    # 未分割のデータを取得
    unsplit_data = TrainData.objects.filter(theme_id=theme_id, split__isnull=True)
    
    if not unsplit_data.exists():
        return get_split_statistics(theme_id)
    
    # ランダムシードを設定
    random.seed(random_seed)
    
    # ラベルごとにデータをグループ化（層化分割）
    label_groups = {}
    for traindata in unsplit_data:
        label_id = traindata.label_id
        if label_id not in label_groups:
            label_groups[label_id] = []
        label_groups[label_id].append(traindata)
    
    # 各ラベルグループを個別に分割
    for label_id, data_list in label_groups.items():
        # ラベルごとにシャッフル
        random.shuffle(data_list)
        
        total = len(data_list)
        train_count = int(total * train_ratio)
        valid_count = int(total * valid_ratio)
        # test_countは残りすべて（丸め誤差対策）
        
        # 分割を割り当て
        for i, traindata in enumerate(data_list):
            if i < train_count:
                traindata.split = 'train'
            elif i < train_count + valid_count:
                traindata.split = 'valid'
            else:
                traindata.split = 'test'
            traindata.save()
    
    return get_split_statistics(theme_id)


def reset_splits(theme_id: int) -> int:
    """
    指定されたテーマの全データの分割情報をリセット
    
    Args:
        theme_id: テーマID
    
    Returns:
        リセットされたデータ数
    """
    count = TrainData.objects.filter(theme_id=theme_id).update(split=None)
    return count


def assign_all_splits(
    theme_id: int,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, int]:
    """
    全データをtrain/valid/testに再分割（層化分割）
    
    既存の分割情報をリセットしてから、全データを再度分割します。
    各クラスの割合がtrain/valid/testで維持されるように分割します。
    
    Args:
        theme_id: テーマID
        train_ratio: 学習データ比率
        valid_ratio: 検証データ比率
        test_ratio: テストデータ比率
        random_seed: ランダムシード
    
    Returns:
        分割結果の統計
    """
    # まず全データの分割をリセット
    reset_splits(theme_id)
    
    # 未分割データとして再分割
    return assign_splits_to_new_data(
        theme_id=theme_id,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

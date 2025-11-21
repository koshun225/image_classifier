"""
ビュー実装

要件定義に基づくLabel-Studio風UI
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.conf import settings
import json
import os
from pathlib import Path

from .models import Theme, Label, TrainData, Model, TrainingJob
from .constants import MLFLOW_UI_URL
from .crud import (
    get_all_themes,
    create_theme,
    get_labels_by_theme,
    create_label,
    delete_label,
    get_traindata_by_theme,
    create_traindata,
    update_traindata_label,
    delete_traindata,
    get_split_statistics,
    assign_splits_to_new_data,
)
from .utils.yaml_utils import (
    load_yaml_file,
    save_yaml_file,
    get_yaml_file_content,
    normalize_params_schema,
    denormalize_params_schema,
    normalize_augments_schema,
    denormalize_augments_schema,
)
from .utils.preview_utils import generate_preprocessing_preview, generate_augmentation_preview
from django.utils import timezone
import subprocess
import tempfile
import yaml
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


def login_view(request):
    """ログイン画面"""
    if request.user.is_authenticated:
        return redirect('theme_list')
    
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('theme_list')
    else:
        form = AuthenticationForm()
    
    return render(request, 'data_management/login.html', {'form': form})


@login_required
def logout_view(request):
    """ログアウト"""
    logout(request)
    return redirect('login')


@login_required
def theme_list(request):
    """テーマ一覧画面"""
    themes = list(get_all_themes())
    
    # 統計情報
    total_themes = len(themes)
    total_images = TrainData.objects.count()
    
    # 各テーマのカウントを確実に取得（annotateの値を使用）
    for theme in themes:
        # annotateで設定された値が存在しない場合は、メソッドで取得
        if not hasattr(theme, 'label_count'):
            theme.label_count = theme.get_label_count()
        if not hasattr(theme, 'image_count'):
            theme.image_count = theme.get_image_count()
    
    context = {
        'themes': themes,
        'total_themes': total_themes,
        'total_images': total_images,
    }
    return render(request, 'data_management/theme_list.html', context)


@login_required
def theme_create(request):
    """テーマ新規作成画面"""
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        description = request.POST.get('description', '').strip()
        
        if not name:
            return render(request, 'data_management/theme_create.html', {
                'error': 'テーマ名は必須です'
            })
        
        # テーマを作成
        theme = create_theme(name=name, description=description)
        
        # ラベルを追加
        labels = request.POST.getlist('labels[]')
        for label_name in labels:
            label_name = label_name.strip()
            if label_name:
                create_label(theme_id=theme.id, label_name=label_name)
        
        return redirect('theme_detail', theme_id=theme.id)
    
    return render(request, 'data_management/theme_create.html')


@login_required
def theme_detail(request, theme_id):
    """テーマ内画面（画像一覧、ラベリング）"""
    theme = get_object_or_404(Theme, id=theme_id)
    labels = get_labels_by_theme(theme_id=theme_id)
    
    # フィルタリング
    filter_type = request.GET.get('filter', 'all')
    filter_label = request.GET.get('label', '')  # ラベルIDでフィルタ
    filter_split = request.GET.get('split', '')  # train/valid/testでフィルタ
    
    queryset = TrainData.objects.filter(theme_id=theme_id)
    
    # 基本フィルタ（未ラベル/ラベル済み）
    if filter_type == 'unlabeled':
        queryset = queryset.filter(label__isnull=True)
    elif filter_type == 'labeled':
        queryset = queryset.filter(label__isnull=False)
    
    # ラベルフィルタ
    if filter_label and filter_label.isdigit():
        queryset = queryset.filter(label_id=int(filter_label))
    
    # 分割フィルタ
    if filter_split:
        if filter_split == 'unsplit':
            queryset = queryset.filter(split__isnull=True)
        elif filter_split in ['train', 'valid', 'test']:
            queryset = queryset.filter(split=filter_split)
    
    # ページネーション
    paginator = Paginator(queryset, 20)  # 1ページ20件
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    
    # 統計情報
    stats = get_split_statistics(theme_id=theme_id)
    
    # ラベルごとの詳細統計
    label_stats = []
    for label in labels:
        train_count = TrainData.objects.filter(theme_id=theme_id, label_id=label.id, split='train').count()
        valid_count = TrainData.objects.filter(theme_id=theme_id, label_id=label.id, split='valid').count()
        test_count = TrainData.objects.filter(theme_id=theme_id, label_id=label.id, split='test').count()
        unsplit_count = TrainData.objects.filter(theme_id=theme_id, label_id=label.id, split__isnull=True).count()
        total_count = train_count + valid_count + test_count + unsplit_count
        
        label_stats.append({
            'label': label,
            'train': train_count,
            'valid': valid_count,
            'test': test_count,
            'unsplit': unsplit_count,
            'total': total_count,
        })
    
    context = {
        'theme': theme,
        'labels': labels,
        'page_obj': page_obj,
        'filter_type': filter_type,
        'filter_label': filter_label,
        'filter_split': filter_split,
        'stats': stats,
        'label_stats': label_stats,
    }
    return render(request, 'data_management/theme_detail.html', context)


# REST APIエンドポイント

@login_required
@require_http_methods(["POST"])
def api_update_label(request, theme_id, traindata_id):
    """ラベル更新API"""
    try:
        data = json.loads(request.body)
        label_id = data.get('label_id')
        
        if label_id:
            label_id = int(label_id)
        else:
            label_id = None
        
        traindata = update_traindata_label(
            theme_id=theme_id,
            traindata_id=traindata_id,
            label_id=label_id
        )
        
        if traindata is None:
            return JsonResponse({'success': False, 'error': '更新に失敗しました'}, status=400)
        
        return JsonResponse({
            'success': True,
            'label_id': traindata.label.id if traindata.label else None,
            'label_name': traindata.label.label_name if traindata.label else None,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required
@require_http_methods(["POST"])
def api_upload_images(request, theme_id):
    """画像アップロードAPI"""
    training_job = None
    try:
        theme = get_object_or_404(Theme, id=theme_id)
        label_id = request.POST.get('label_id')
        
        if label_id:
            label_id = int(label_id)
            label = get_object_or_404(Label, id=label_id, theme_id=theme_id)
        else:
            label = None
        
        uploaded_files = request.FILES.getlist('images')
        created_count = 0
        
        for image_file in uploaded_files:
            create_traindata(
                theme_id=theme_id,
                image=image_file,
                label_id=label_id,
                labeled_by=request.user.username
            )
            created_count += 1
        
        return JsonResponse({
            'success': True,
            'created_count': created_count,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required
@require_http_methods(["DELETE"])
def api_delete_image(request, theme_id, traindata_id):
    """画像削除API"""
    try:
        success = delete_traindata(theme_id=theme_id, traindata_id=traindata_id)
        
        if not success:
            return JsonResponse({'success': False, 'error': '削除に失敗しました'}, status=400)
        
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required
@require_http_methods(["POST"])
def api_split_data(request, theme_id):
    """データ分割API"""
    try:
        data = json.loads(request.body)
        train_ratio = float(data.get('train_ratio', 0.7))
        valid_ratio = float(data.get('valid_ratio', 0.15))
        test_ratio = float(data.get('test_ratio', 0.15))
        random_seed = int(data.get('random_seed', 42))
        unsplit_only = bool(data.get('unsplit_only', True))  # デフォルトはTrue（未分割のみ）
        
        # unsplit_onlyがTrueの場合は未分割データのみを分割、Falseの場合は全データを再分割
        if unsplit_only:
            stats = assign_splits_to_new_data(
                theme_id=theme_id,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed
            )
        else:
            from data_management.crud import assign_all_splits
            stats = assign_all_splits(
                theme_id=theme_id,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed
            )
        
        return JsonResponse({
            'success': True,
            'stats': stats,
            'unsplit_only': unsplit_only,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required
@require_http_methods(["GET"])
def api_statistics(request, theme_id):
    """統計取得API"""
    try:
        stats = get_split_statistics(theme_id=theme_id)
        
        # ラベル別統計
        label_stats = TrainData.objects.filter(theme_id=theme_id).values(
            'label__id', 'label__label_name'
        ).annotate(count=Count('id')).order_by('label__id')
        
        label_counts = [
            {
                'label_id': item['label__id'],
                'label_name': item['label__label_name'],
                'count': item['count']
            }
            for item in label_stats if item['label__id']
        ]
        
        return JsonResponse({
            'success': True,
            'stats': stats,
            'label_counts': label_counts,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


def _get_mlflow_tracking_uri() -> str:
    """
    MLflowのtracking URIをconfig.yamlから取得
    """
    default_uri = "experiments/mlruns"
    try:
        project_root = Path(settings.BASE_DIR).resolve().parent.parent
        config_path = project_root / "config.yaml"
        if not config_path.exists():
            return default_uri
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        return config.get("mlflow", {}).get("tracking_uri", default_uri)
    except Exception:
        return default_uri


# モデル開発関連ビュー

@login_required
def model_development(request, theme_id):
    """パラメータ設定画面"""
    theme = get_object_or_404(Theme, id=theme_id)
    
    # 再学習の場合、モデルのパラメータを読み込む
    model_id = request.GET.get('model_id')
    if model_id:
        try:
            model = Model.objects.get(id=model_id, theme_id=theme_id)
            training_params = model.get_training_params_dict()
            if training_params:
                # チューニング対象を削除して固定値のみにする
                from .utils.yaml_utils import remove_tunable_specs
                # training_paramsは既にdenormalizeされた形式（単純な値）である可能性がある
                # remove_tunable_specsはtypeプロパティを持つノードからvalueのみを抽出する
                params_schema = remove_tunable_specs(training_params)
                # optunaセクションを追加（n_trials=1でチューニングなし）
                params_schema['optuna'] = {
                    'metric': 'test_acc',
                    'direction': 'maximize',
                    'n_trials': 1,
                    'timeout': None
                }
            else:
                params_schema = load_yaml_file('params.yaml') or {}
        except Model.DoesNotExist:
            params_schema = load_yaml_file('params.yaml') or {}
    else:
        params_schema = load_yaml_file('params.yaml') or {}
    
    optuna_config = params_schema.get('optuna', {})
    
    # data.theme_idをURLから取得したtheme_idで上書き（読み取り専用）
    if 'data' not in params_schema:
        params_schema['data'] = {}
    params_schema['data']['theme_id'] = theme_id
    
    # model.nameの選択肢をMODEL_REGISTRYから取得したものに更新
    from .utils.yaml_utils import get_available_models
    available_models = get_available_models()
    if 'model' not in params_schema:
        params_schema['model'] = {}
    
    # 再学習時はtypeを設定しない（チューニング対象にしない）
    is_retrain = bool(model_id)
    print(params_schema, 'params_schema')
    # model.nameを正しい形式に変換
    if 'name' not in params_schema['model']:
        model_name_value = available_models[0] if available_models else 'ResNet18'
    else:
        model_name_value = params_schema['model']['name']
    
    if isinstance(model_name_value, str):
        # 固定値の場合
        if is_retrain:
            # 再学習時はtypeを設定しない
            params_schema['model']['name'] = {'value': model_name_value}
        else:
            params_schema['model']['name'] = {
                'value': model_name_value,
                'type': 'categorical',
                'choices': available_models
            }
    elif isinstance(model_name_value, dict):
        # 既にdict形式の場合
        if 'type' in model_name_value and not is_retrain:
            # 通常時はtypeを保持し、選択肢を更新
            params_schema['model']['name']['choices'] = available_models
            current_value = model_name_value.get('value')
            if current_value not in available_models:
                params_schema['model']['name']['value'] = available_models[0] if available_models else 'ResNet18'
        elif 'type' in model_name_value and is_retrain:
            # 再学習時はtypeを削除
            params_schema['model']['name'] = {'value': model_name_value.get('value', model_name_value)}
        elif 'value' not in model_name_value:
            # valueがない場合は、dict全体をvalueとして扱う
            if is_retrain:
                params_schema['model']['name'] = {'value': model_name_value}
            else:
                params_schema['model']['name'] = {
                    'value': model_name_value,
                    'type': 'categorical',
                    'choices': available_models
                }
        elif is_retrain:
            # 再学習時はtypeを削除
            params_schema['model']['name'] = {'value': model_name_value.get('value')}
    
    # optimizerの選択肢を確認・設定（Adam, SGD, AdamW）
    if 'training' not in params_schema:
        params_schema['training'] = {}
    
    optimizer_value = params_schema['training'].get('optimizer')

    # remove_tunable_specsの結果は単純な値（str）になっているはず
    if optimizer_value is None:
        if is_retrain:
            params_schema['training']['optimizer'] = {'value': 'Adam'}
        else:
            params_schema['training']['optimizer'] = {
                'value': 'Adam',
                'type': 'categorical',
                'choices': ['Adam', 'SGD', 'AdamW']
            }
    elif isinstance(optimizer_value, str):
        # remove_tunable_specsの結果は単純な値（str）になっている
        if is_retrain:
            params_schema['training']['optimizer'] = {'value': optimizer_value}
        else:
            params_schema['training']['optimizer'] = {
                'value': optimizer_value,
                'type': 'categorical',
                'choices': ['Adam', 'SGD', 'AdamW']
            }
    elif isinstance(optimizer_value, dict):
        # dict形式の場合（通常時は発生しないが、念のため）
        if 'type' in optimizer_value and not is_retrain:
            # 通常時はtypeを保持し、選択肢を更新
            optimizer_choices = optimizer_value.get('choices', [])
            valid_choices = ['Adam', 'SGD', 'AdamW']
            if set(optimizer_choices) != set(valid_choices):
                params_schema['training']['optimizer']['choices'] = valid_choices
            current_optimizer = optimizer_value.get('value')
            if current_optimizer not in valid_choices:
                params_schema['training']['optimizer']['value'] = 'Adam'
        elif 'type' in optimizer_value and is_retrain:
            # 再学習時はtypeを削除
            params_schema['training']['optimizer'] = {'value': optimizer_value.get('value')}
        elif 'value' in optimizer_value:
            # valueがある場合はvalueのみを使用
            if is_retrain:
                params_schema['training']['optimizer'] = {'value': optimizer_value.get('value')}
            else:
                params_schema['training']['optimizer'] = {
                    'value': optimizer_value.get('value'),
                    'type': 'categorical',
                    'choices': ['Adam', 'SGD', 'AdamW']
                }
        else:
            # valueがない場合は、dict全体をvalueとして扱う
            if is_retrain:
                params_schema['training']['optimizer'] = {'value': optimizer_value}
            else:
                params_schema['training']['optimizer'] = {
                    'value': optimizer_value,
                    'type': 'categorical',
                    'choices': ['Adam', 'SGD', 'AdamW']
                }
    
    # num_classesをテーマのラベル数から自動取得（読み取り専用）
    num_classes = theme.get_label_count()
    if 'model' not in params_schema:
        params_schema['model'] = {}
    params_schema['model']['num_classes'] = {
        'value': num_classes,
        'readonly': True  # 読み取り専用フラグ
    }
    
    # normalize_params_schemaを呼び出す前に、params_schemaが正しい形式になっているか確認
    # training_paramsがdenormalizeされた形式（単純な値）の場合、normalize_params_schemaで正しく処理される
    # ただし、remove_tunable_specsの結果は単純な値になっているので、
    # normalize_params_schemaが{'value': ...}形式に変換する前に、
    # すべてのパラメータを正しい形式に変換する必要がある
    params_form_data = normalize_params_schema(params_schema)
    params_form_data.pop('optuna', None)
    
    # 再学習時は、normalize_params_schemaの結果を確認し、
    # 必要に応じてパラメータを修正する
    if is_retrain:
        # optimizerが正しく設定されているか確認
        if params_form_data.get('training', {}).get('optimizer'):
            optimizer_node = params_form_data['training']['optimizer']
            if isinstance(optimizer_node, dict) and 'value' in optimizer_node:
                # 既に正しい形式
                pass
            elif isinstance(optimizer_node, str):
                # 単純な値の場合は{'value': ...}形式に変換
                params_form_data['training']['optimizer'] = {'value': optimizer_node}
        
        # learning_rateが正しく設定されているか確認
        if params_form_data.get('training', {}).get('learning_rate'):
            lr_node = params_form_data['training']['learning_rate']
            if isinstance(lr_node, dict) and 'value' in lr_node:
                # 既に正しい形式
                pass
            elif not isinstance(lr_node, dict):
                # 単純な値の場合は{'value': ...}形式に変換
                params_form_data['training']['learning_rate'] = {'value': lr_node}
        
        # batch_sizeが正しく設定されているか確認
        if params_form_data.get('training', {}).get('batch_size'):
            bs_node = params_form_data['training']['batch_size']
            if isinstance(bs_node, dict) and 'value' in bs_node:
                # 既に正しい形式
                pass
            elif not isinstance(bs_node, dict):
                # 単純な値の場合は{'value': ...}形式に変換
                params_form_data['training']['batch_size'] = {'value': bs_node}
        
        # その他のtrainingパラメータも同様に処理
        if 'training' in params_form_data:
            for key, value in params_form_data['training'].items():
                if key in ['optimizer', 'learning_rate', 'batch_size', 'num_epochs', 'num_workers', 'run_name', 'scheduler', 'scheduler_params']:
                    if not isinstance(value, dict):
                        params_form_data['training'][key] = {'value': value}
                    elif 'value' not in value:
                        # valueがない場合は、dict全体をvalueとして扱う
                        params_form_data['training'][key] = {'value': value}
    auguments_schema = load_yaml_file('auguments.yaml') or {}
    auguments_form_data = normalize_augments_schema(auguments_schema)
    
    # 登録済みモデルのリストを取得
    registered_models = Model.objects.filter(theme_id=theme_id, status='completed').order_by('-created_at')
    models_list = [
        {
            'id': model.id,
            'mlflow_run_id': model.mlflow_run_id,
            'best_score': model.best_score,
            'created_at': model.created_at.strftime('%Y-%m-%d %H:%M') if model.created_at else '',
            'created_by': model.created_by or '',
        }
        for model in registered_models
    ]
    
    context = {
        'theme': theme,
        'theme_id': theme_id,
        'available_models': json.dumps(available_models, ensure_ascii=False),
        'params_json': json.dumps(params_form_data, ensure_ascii=False),
        'optuna_json': json.dumps(optuna_config or {}, ensure_ascii=False),
        'auguments_json': json.dumps(auguments_form_data, ensure_ascii=False),
        'registered_models': json.dumps(models_list, ensure_ascii=False),
        'selected_model_id': model_id,  # クエリパラメータから取得したmodel_id
    }
    return render(request, 'data_management/model_development.html', context)


@login_required
@require_http_methods(["GET"])
def api_get_model_params(request, theme_id, model_id):
    """モデルのパラメータとチェックポイントパスを取得するAPI"""
    try:
        model = get_object_or_404(Model, id=model_id, theme_id=theme_id)
        training_params = model.get_training_params_dict()
        
        # MLflowからチェックポイントパスを取得
        checkpoint_path = None
        try:
            from mlflow.tracking import MlflowClient
            from .constants import MLFLOW_UI_URL
            import os
            
            # MLflow Tracking URIを取得
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5001')
            client = MlflowClient(tracking_uri=tracking_uri)
            
            # Run情報を取得
            run = client.get_run(model.mlflow_run_id)
            
            # チェックポイントパスを取得（artifact_path="model"のパス）
            artifact_root = run.info.artifact_uri
            # artifact_uriは通常 "file:///path/to/artifacts" または "mlflow-artifacts:/..." の形式
            # 実際のファイルパスに変換
            if artifact_root.startswith('file://'):
                artifact_root = artifact_root[7:]
            elif artifact_root.startswith('mlflow-artifacts:'):
                # MLflow Tracking Serverを使用している場合
                # experiments/mlruns/{experiment_id}/{run_id}/artifacts の形式
                checkpoint_path = f"{artifact_root}/model"
            else:
                checkpoint_path = f"{artifact_root}/model"
            
            # 実際のファイルパスを構築
            # MLflowのartifactは通常 experiments/mlruns/{experiment_id}/{run_id}/artifacts/model/ に保存される
            if not checkpoint_path.startswith('mlflow-artifacts:'):
                # ファイルシステムパスの場合
                checkpoint_path = os.path.join(artifact_root, 'model')
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"MLflowからチェックポイントパスを取得できませんでした: {e}")
            checkpoint_path = None
        
        # パラメータを正規化
        from .utils.yaml_utils import remove_tunable_specs, normalize_params_schema
        params_schema = remove_tunable_specs(training_params) if training_params else {}
        params_form_data = normalize_params_schema(params_schema)
        params_form_data.pop('optuna', None)
        
        return JsonResponse({
            'success': True,
            'params': params_form_data,
            'checkpoint_path': checkpoint_path,
            'mlflow_run_id': model.mlflow_run_id,
        })
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"モデルパラメータ取得エラー: {e}", exc_info=True)
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required
@require_http_methods(["POST"])
def api_save_params(request, theme_id):
    """パラメータ保存API"""
    try:
        data = json.loads(request.body)
        params_schema_data = data.get('params_schema')
        params_content = data.get('params', '')
        optuna_config_data = data.get('optuna_config') or {}
        auguments_content = data.get('auguments', '')
        
        # YAMLバリデーション
        try:
            yaml.safe_load(auguments_content)
        except yaml.YAMLError as e:
            return JsonResponse({'success': False, 'error': f'YAML形式エラー: {str(e)}'}, status=400)
        
        # params.yaml
        if params_schema_data is not None:
            theme = get_object_or_404(Theme, id=theme_id)
            params_payload = denormalize_params_schema(params_schema_data)
            # data.theme_idをURLから取得したtheme_idで強制的に上書き（変更不可）
            if 'data' not in params_payload:
                params_payload['data'] = {}
            params_payload['data']['theme_id'] = theme_id
            # num_classesをテーマのラベル数で強制的に上書き（変更不可）
            if 'model' not in params_payload:
                params_payload['model'] = {}
            params_payload['model']['num_classes'] = theme.get_label_count()
            if optuna_config_data is not None:
                params_payload['optuna'] = optuna_config_data
            elif 'optuna' not in params_payload:
                params_payload['optuna'] = {}
            save_yaml_file('params.yaml', params_payload)
        elif params_content:
            save_yaml_file('params.yaml', yaml.safe_load(params_content))
        else:
            return JsonResponse({'success': False, 'error': 'paramsデータが提供されていません'}, status=400)

        # auguments.yaml
        auguments_schema_data = data.get('auguments_schema')
        if auguments_schema_data is not None:
            auguments_payload = denormalize_augments_schema(auguments_schema_data)
            save_yaml_file('auguments.yaml', auguments_payload)
        elif auguments_content:
            save_yaml_file('auguments.yaml', yaml.safe_load(auguments_content) if auguments_content else {})
        else:
            save_yaml_file('auguments.yaml', {})
        
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required
@require_http_methods(["POST"])
def api_preview_augmentation(request, theme_id):
    """augmentation/preprocessingプレビュー生成API"""
    try:
        data = json.loads(request.body)
        auguments_yaml = data.get('auguments_yaml', '')
        auguments_schema_data = data.get('auguments_schema')
        preview_type = data.get('type', 'both')  # 'preprocessing', 'augmentation', 'both'
        
        # YAMLをパース
        if auguments_schema_data is not None:
            auguments_config = denormalize_augments_schema(auguments_schema_data)
        elif auguments_yaml:
            auguments_config = yaml.safe_load(auguments_yaml)
        else:
            auguments_config = {}
        
        result = {}
        
        if preview_type in ['preprocessing', 'both']:
            preprocessing_result = generate_preprocessing_preview(theme_id, auguments_config, num_images=5)
            result['preprocessing'] = preprocessing_result
        
        if preview_type in ['augmentation', 'both']:
            augmentation_result = generate_augmentation_preview(theme_id, auguments_config, num_images=5, num_samples=5)
            result['augmentation'] = augmentation_result
        
        return JsonResponse({'success': True, 'preview': result})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required
@require_http_methods(["POST"])
def api_start_training(request, theme_id):
    """学習開始API"""
    try:
        theme = get_object_or_404(Theme, id=theme_id)
        
        # リクエストボディからチェックポイントパスを取得
        data = json.loads(request.body) if request.body else {}
        checkpoint_path = data.get('checkpoint_path')
        
        # 既存の実行中ジョブをチェック（実際にプロセスが存在するかも確認）
        running_jobs = TrainingJob.objects.filter(theme_id=theme_id, status='running')
        for job in running_jobs:
            # プロセスが実際に存在するか確認
            if job.process_id:
                try:
                    os.kill(job.process_id, 0)  # プロセスが存在するかチェック
                    # プロセスが存在する場合は実行中
                    return JsonResponse({
                        'success': False,
                        'error': '既に実行中のジョブがあります'
                    }, status=400)
                except (OSError, ProcessLookupError):
                    # プロセスが存在しない場合は、ステータスを更新
                    job.status = 'failed'
                    job.error_message = 'プロセスが異常終了しました（自動検出）'
                    job.completed_at = timezone.now()
                    job.save()
            else:
                # プロセスIDがない場合は、ステータスを更新
                job.status = 'failed'
                job.error_message = 'プロセスIDが設定されていません'
                job.completed_at = timezone.now()
                job.save()
        
        # YAMLファイルの内容を取得して保存
        params_content = get_yaml_file_content('params.yaml')
        params_dict = yaml.safe_load(params_content) if params_content else {}
        
        # チェックポイントパスをparams.yamlに追加
        if checkpoint_path:
            if 'training' not in params_dict:
                params_dict['training'] = {}
            params_dict['training']['checkpoint_path'] = checkpoint_path
            # params.yamlを更新
            save_yaml_file('params.yaml', params_dict)
            params_content = get_yaml_file_content('params.yaml')
        
        optuna_section = (params_dict or {}).get('optuna', {})
        optuna_params_content = yaml.safe_dump(optuna_section, allow_unicode=True)
        auguments_content = get_yaml_file_content('auguments.yaml')
        
        # TrainingJobを作成
        log_dir = tempfile.gettempdir()
        log_file = os.path.join(log_dir, f'training_job_{theme_id}_{timezone.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        training_job = TrainingJob.objects.create(
            theme=theme,
            status='pending',
            log_file=log_file,
            params_yaml=params_content,
            optuna_params_yaml=optuna_params_content,
            auguments_yaml=auguments_content,
            created_by=request.user.username,
        )
        
        # scripts/tune.pyをバックグラウンドで実行
        # views.py は src/web/data_management/views.py なので、4階層上がる
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        script_path = project_root / 'scripts' / 'tune.py'
        
        # スクリプトファイルの存在確認
        if not script_path.exists():
            return JsonResponse({
                'success': False,
                'error': f'スクリプトファイルが見つかりません: {script_path}'
            }, status=400)
        
        # ログファイルを開く
        log_file_handle = open(log_file, 'w')
        
        # プロセスを開始
        process = subprocess.Popen(
            ['python', str(script_path), '--theme-id', str(theme_id), '--training-job-id', str(training_job.id)],
            stdout=log_file_handle,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            start_new_session=True
        )
        
        # プロセスIDとステータスを更新
        training_job.process_id = process.pid
        training_job.status = 'running'
        training_job.started_at = timezone.now()
        training_job.save()
        
        log_file_handle.close()
        
        return JsonResponse({
            'success': True,
            'job_id': training_job.id,
        })
    except Exception as e:
        # ジョブが作成済みの場合はステータスを失敗に更新
        if training_job is not None:
            training_job.status = 'failed'
            training_job.error_message = str(e)
            training_job.completed_at = timezone.now()
            training_job.save()
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required
@require_http_methods(["GET"])
def api_training_status(request, theme_id, job_id):
    """学習進捗状況取得API"""
    try:
        training_job = get_object_or_404(TrainingJob, id=job_id, theme_id=theme_id)
        
        # ログファイルの内容を取得（最後の100行）
        log_content = ""
        if training_job.log_file and os.path.exists(training_job.log_file):
            with open(training_job.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                log_content = ''.join(lines[-100:])  # 最後の100行
        
        # プロセスの状態を確認
        if training_job.process_id:
            try:
                os.kill(training_job.process_id, 0)  # プロセスが存在するかチェック
                is_running = True
            except (OSError, ProcessLookupError):
                is_running = False
                if training_job.status == 'running':
                    training_job.status = 'failed'
                    training_job.error_message = 'プロセスが異常終了しました'
                    training_job.completed_at = timezone.now()
                    training_job.save()
        else:
            is_running = False
        
        return JsonResponse({
            'success': True,
            'status': training_job.status,
            'log': log_content,
            'started_at': training_job.started_at.isoformat() if training_job.started_at else None,
            'completed_at': training_job.completed_at.isoformat() if training_job.completed_at else None,
            'is_running': is_running,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required
def model_training_status(request, theme_id, job_id):
    """学習実行・進捗画面"""
    theme = get_object_or_404(Theme, id=theme_id)
    training_job = get_object_or_404(TrainingJob, id=job_id, theme_id=theme_id)
    
    # MLflow UI URL（constants.pyから取得）
    mlflow_ui_url = MLFLOW_UI_URL

    mlflow_experiment_id = None
    mlflow_run_id = None
    if training_job.model and training_job.model.mlflow_run_id:
        mlflow_run_id = training_job.model.mlflow_run_id
    elif training_job.mlflow_parent_run_id:
        mlflow_run_id = training_job.mlflow_parent_run_id

    # experiment_idを取得（mlflow_run_idがなくてもテーマ名から取得可能）
    tracking_uri = _get_mlflow_tracking_uri()
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        # まずはテーマ名のExperimentを取得（訓練時と同じ命名）
        experiment = client.get_experiment_by_name(theme.name)
        if experiment:
            mlflow_experiment_id = experiment.experiment_id
        elif mlflow_run_id:
            # fallback: run情報からexperiment_idを取得
            run = client.get_run(mlflow_run_id)
            mlflow_experiment_id = run.info.experiment_id
    except MlflowException:
        mlflow_experiment_id = None
    
    context = {
        'theme': theme,
        'training_job': training_job,
        'mlflow_ui_url': mlflow_ui_url,
        'mlflow_experiment_id': mlflow_experiment_id,
        'mlflow_run_id': mlflow_run_id,
    }
    return render(request, 'data_management/model_training_status.html', context)


@login_required
def model_list(request, theme_id):
    """モデル一覧画面"""
    theme = get_object_or_404(Theme, id=theme_id)
    models = Model.objects.filter(theme_id=theme_id).order_by('-created_at')
    

    mlflow_experiment_id = None
    tracking_uri = _get_mlflow_tracking_uri()
    client = MlflowClient(tracking_uri=tracking_uri)
    print(tracking_uri, "tracking_uri")
    try:
        experiment = client.get_experiment_by_name(theme.name)
        
        if experiment:
            mlflow_experiment_id = experiment.experiment_id
    except MlflowException:
        mlflow_experiment_id = None
    
    context = {
        'theme': theme,
        'models': models,
        'mlflow_ui_url': tracking_uri,
        'mlflow_experiment_id': mlflow_experiment_id,
    }
    return render(request, 'data_management/model_list.html', context)

// テーマ詳細画面のJavaScript
console.log('theme_detail.js loaded');

let selectedLabelId = null;

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded fired');
    
    // 画像カードクリックでラベル選択モーダルを開く
    document.querySelectorAll('.image-card').forEach(function(card) {
        card.addEventListener('click', function(e) {
            // 削除ボタンやその親要素がクリックされた場合は処理しない
            if (e.target.classList.contains('btn-image-delete') || 
                e.target.closest('.btn-image-delete') ||
                e.target.closest('.image-overlay')) {
                return;
            }
            
            const imageId = this.dataset.imageId;
            if (!imageId) {
                console.error('Image ID not found');
                return;
            }
            
            // モーダルで画像を表示してラベル選択
            showImageModal(imageId);
        });
    });
    
    // 画像削除（カード）
    document.querySelectorAll('.btn-image-delete').forEach(function(btn) {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const imageId = this.dataset.imageId;
            if (confirm('この画像を削除しますか？')) {
                deleteImage(imageId);
            }
        });
    });
    
    // 表示形式の切り替え
    const viewGridBtn = document.getElementById('view-grid-btn');
    const viewTableBtn = document.getElementById('view-table-btn');
    const imageGrid = document.getElementById('image-grid');
    const imageTable = document.getElementById('image-table');
    
    console.log('View toggle elements:', { viewGridBtn, viewTableBtn, imageGrid, imageTable });
    
    // localStorageから表示形式を取得
    const savedView = localStorage.getItem('imageViewMode') || 'grid';
    console.log('Saved view mode:', savedView);
    
    function showGridView() {
        console.log('Switching to grid view');
        if (imageGrid) imageGrid.style.display = 'grid';
        if (imageTable) imageTable.style.display = 'none';
        if (viewGridBtn) viewGridBtn.classList.add('active');
        if (viewTableBtn) viewTableBtn.classList.remove('active');
    }
    
    function showTableView() {
        console.log('Switching to table view');
        if (imageGrid) imageGrid.style.display = 'none';
        if (imageTable) imageTable.style.display = 'block';
        if (viewGridBtn) viewGridBtn.classList.remove('active');
        if (viewTableBtn) viewTableBtn.classList.add('active');
    }
    
    if (savedView === 'table' && imageTable) {
        showTableView();
    } else if (imageGrid) {
        showGridView();
    }
    
    if (viewGridBtn && viewTableBtn) {
        console.log('Adding click listeners to view buttons');
        viewGridBtn.addEventListener('click', function(e) {
            console.log('Grid button clicked');
            e.preventDefault();
            showGridView();
            localStorage.setItem('imageViewMode', 'grid');
        });
        
        viewTableBtn.addEventListener('click', function(e) {
            console.log('Table button clicked');
            e.preventDefault();
            showTableView();
            localStorage.setItem('imageViewMode', 'table');
        });
    } else {
        console.error('View toggle buttons not found!', { viewGridBtn, viewTableBtn });
    }
    
    // テーブル内の画像行クリックでモーダル表示
    document.querySelectorAll('.image-row').forEach(function(row) {
        row.addEventListener('click', function(e) {
            // ボタンクリックは除外
            if (e.target.classList.contains('btn-table-label') || 
                e.target.classList.contains('btn-table-delete') ||
                e.target.closest('.btn-table-label') ||
                e.target.closest('.btn-table-delete')) {
                return;
            }
            
            const imageId = this.dataset.imageId;
            if (imageId) {
                showImageModal(imageId);
            }
        });
    });
    
    // テーブル内のラベル変更ボタン
    document.querySelectorAll('.btn-table-label').forEach(function(btn) {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const imageId = this.dataset.imageId;
            if (imageId) {
                showImageModal(imageId);
            }
        });
    });
    
    // テーブル内の削除ボタン
    document.querySelectorAll('.btn-table-delete').forEach(function(btn) {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const imageId = this.dataset.imageId;
            if (confirm('この画像を削除しますか？')) {
                deleteImage(imageId);
            }
        });
    });
    
    // 画像アップロード
    const uploadBtn = document.getElementById('upload-btn');
    const uploadModal = document.getElementById('upload-modal');
    const uploadForm = document.getElementById('upload-form');
    
    if (uploadBtn) {
        uploadBtn.addEventListener('click', function() {
            uploadModal.style.display = 'block';
        });
    }
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            uploadImages();
        });
    }
    
    // データ分割
    const splitBtn = document.getElementById('split-btn');
    const splitModal = document.getElementById('split-modal');
    const splitForm = document.getElementById('split-form');
    
    if (splitBtn) {
        splitBtn.addEventListener('click', function() {
            splitModal.style.display = 'block';
        });
    }
    
    if (splitForm) {
        splitForm.addEventListener('submit', function(e) {
            e.preventDefault();
            splitData();
        });
    }
    
    // フィルターボタン（表示: すべて/未ラベル/ラベル済み）
    document.querySelectorAll('.filter-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            const filterType = this.dataset.filterType;
            const url = new URL(window.location);
            url.searchParams.set('filter', filterType);
            url.searchParams.set('page', '1');
            window.location.href = url.toString();
        });
    });
    
    // ラベルフィルタ（セレクトボックス）
    const filterLabelSelect = document.getElementById('filter-label');
    if (filterLabelSelect) {
        filterLabelSelect.addEventListener('change', function() {
            const labelId = this.value;
            const url = new URL(window.location);
            
            if (labelId) {
                url.searchParams.set('label', labelId);
            } else {
                url.searchParams.delete('label');
            }
            url.searchParams.set('page', '1');
            window.location.href = url.toString();
        });
    }
    
    // 分割フィルタボタン（Train/Valid/Test/未分割）
    document.querySelectorAll('.filter-split-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            const splitValue = this.dataset.filterSplit;
            const url = new URL(window.location);
            
            if (splitValue) {
                url.searchParams.set('split', splitValue);
            } else {
                url.searchParams.delete('split');
            }
            url.searchParams.set('page', '1');
            window.location.href = url.toString();
        });
    });
    
    // フィルタクリアボタン
    const clearFiltersBtn = document.getElementById('clear-filters-btn');
    if (clearFiltersBtn) {
        clearFiltersBtn.addEventListener('click', function() {
            const url = new URL(window.location);
            url.searchParams.delete('filter');
            url.searchParams.delete('label');
            url.searchParams.delete('split');
            url.searchParams.set('page', '1');
            window.location.href = url.toString();
        });
    }
    
    // 統計情報を定期的に更新
    setInterval(updateStatistics, 10000);
});

// ラベル更新
function updateLabel(imageId, labelId) {
    if (!imageId || !labelId) {
        console.error('Image ID or Label ID is missing', { imageId, labelId });
        alert('画像IDまたはラベルIDが正しくありません');
        return;
    }
    
    const url = `/api/theme/${themeId}/label/update/${imageId}/`;
    
    console.log('Updating label:', { imageId, labelId, url });
    
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({ label_id: labelId })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'HTTP error ' + response.status);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // ページをリロードして更新
            location.reload();
        } else {
            alert('ラベルの更新に失敗しました: ' + (data.error || '不明なエラー'));
        }
    })
    .catch(error => {
        console.error('Error updating label:', error);
        alert('エラーが発生しました: ' + error.message);
    });
}

// 画像削除
function deleteImage(imageId) {
    const url = `/api/theme/${themeId}/images/${imageId}/`;
    
    fetch(url, {
        method: 'DELETE',
        headers: {
            'X-CSRFToken': csrfToken
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // ページをリロード
            location.reload();
        } else {
            alert('画像の削除に失敗しました: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('エラーが発生しました');
    });
}

// 画像アップロード
function uploadImages() {
    const form = document.getElementById('upload-form');
    const formData = new FormData(form);
    
    const url = `/api/theme/${themeId}/images/upload/`;
    
    fetch(url, {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrfToken
        },
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(`${data.created_count}件の画像をアップロードしました`);
            document.getElementById('upload-modal').style.display = 'none';
            location.reload();
        } else {
            alert('画像のアップロードに失敗しました: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('エラーが発生しました');
    });
}

// データ分割
function splitData() {
    const form = document.getElementById('split-form');
    const formData = new FormData(form);
    const unsplitOnlyCheckbox = document.getElementById('unsplit-only');
    
    const data = {
        train_ratio: parseFloat(formData.get('train_ratio')),
        valid_ratio: parseFloat(formData.get('valid_ratio')),
        test_ratio: parseFloat(formData.get('test_ratio')),
        random_seed: parseInt(formData.get('random_seed')),
        unsplit_only: unsplitOnlyCheckbox.checked
    };
    
    // 全データ再分割の場合は確認メッセージを表示
    if (!data.unsplit_only) {
        const confirmed = confirm(
            '⚠️ 警告：全データを再分割します。\n\n' +
            '既存の分割情報が削除され、学習済みモデルとの整合性が失われる可能性があります。\n\n' +
            '本当に実行しますか？'
        );
        if (!confirmed) {
            return;
        }
    }
    
    const url = `/api/theme/${themeId}/split/`;
    
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const message = data.unsplit_only 
                ? 'データ分割が完了しました（未分割データのみ）' 
                : 'データ分割が完了しました（全データを再分割）';
            alert(message);
            document.getElementById('split-modal').style.display = 'none';
            updateStatistics();
            location.reload();
        } else {
            alert('データ分割に失敗しました: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('エラーが発生しました');
    });
}

// 統計情報更新
function updateStatistics() {
    const url = `/api/theme/${themeId}/statistics/`;
    
    fetch(url, {
        method: 'GET',
        headers: {
            'X-CSRFToken': csrfToken
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const stats = data.stats;
            document.getElementById('train-count').textContent = stats.train || 0;
            document.getElementById('valid-count').textContent = stats.valid || 0;
            document.getElementById('test-count').textContent = stats.test || 0;
            document.getElementById('unsplit-count').textContent = stats.unsplit || 0;
            
            const total = (stats.train || 0) + (stats.valid || 0) + (stats.test || 0) + (stats.unsplit || 0);
            document.getElementById('total-count').textContent = total;
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// 画像モーダル表示
let currentKeyHandler = null;

function showImageModal(imageId) {
    const modal = document.getElementById('image-modal');
    const modalContent = document.getElementById('image-modal-content');
    
    if (!modal || !modalContent) {
        console.error('Image modal not found');
        return;
    }
    
    // 既存のキーボードハンドラーを削除
    if (currentKeyHandler) {
        document.removeEventListener('keydown', currentKeyHandler);
        currentKeyHandler = null;
    }
    
    // 画像情報を取得
    const imageCard = document.querySelector(`.image-card[data-image-id="${imageId}"]`);
    if (!imageCard) {
        console.error('Image card not found:', imageId);
        return;
    }
    
    const imageSrc = imageCard.querySelector('.image-thumbnail').src;
    const currentLabel = imageCard.querySelector('.label-badge')?.textContent || '未ラベル';
    
    // モーダルコンテンツを設定
    modalContent.innerHTML = `
        <div class="image-modal-header">
            <h2>ラベルを選択</h2>
        </div>
        <div class="image-modal-image">
            <img src="${imageSrc}" alt="画像">
        </div>
        <div class="image-modal-label-info">
            <p>現在のラベル: <strong>${currentLabel}</strong></p>
        </div>
        <div class="image-modal-labels">
            <h3>ラベルを選択</h3>
            <div class="label-list-modal" id="label-list-modal">
                ${labels.length > 0 ? 
                    labels.map((label, index) => `
                        <button class="label-btn-modal" 
                                data-label-id="${label.id}" 
                                data-label-name="${label.name}">
                            <span class="label-number">${index + 1}</span>
                            <span class="label-name">${label.name}</span>
                        </button>
                    `).join('') : 
                    '<p>ラベルがありません</p>'
                }
                <button class="label-btn-modal label-btn-unlabel" data-label-id="">
                    <span class="label-name">未ラベルにする (0キー)</span>
                </button>
            </div>
        </div>
        <div class="image-modal-actions">
            <button class="btn btn-secondary modal-close-btn">キャンセル (Esc)</button>
        </div>
    `;
    
    // モーダルを表示
    modal.style.display = 'block';
    
    // ラベルボタンのイベントリスナーを設定
    modalContent.querySelectorAll('.label-btn-modal').forEach(function(btn) {
        btn.addEventListener('click', function() {
            const labelId = this.dataset.labelId || null;
            const labelName = this.dataset.labelName || '未ラベル';
            
            // ラベルを更新
            updateLabel(imageId, labelId);
            
            // モーダルを閉じる
            closeImageModal();
        });
    });
    
    // モーダルを閉じる関数
    const closeImageModal = function() {
        modal.style.display = 'none';
        if (currentKeyHandler) {
            document.removeEventListener('keydown', currentKeyHandler);
            currentKeyHandler = null;
        }
    };
    
    // キーボードショートカット（モーダル内）
    const handleKeyPress = function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        const key = e.key;
        if (key >= '1' && key <= '9') {
            const index = parseInt(key) - 1;
            const labelBtn = modalContent.querySelectorAll('.label-btn-modal')[index];
            if (labelBtn && !labelBtn.classList.contains('label-btn-unlabel')) {
                labelBtn.click();
            }
        } else if (key === '0') {
            // 0キーで未ラベルにする
            const unlabelBtn = modalContent.querySelector('.label-btn-unlabel');
            if (unlabelBtn) {
                unlabelBtn.click();
            }
        } else if (key === 'Escape') {
            closeImageModal();
        }
    };
    
    currentKeyHandler = handleKeyPress;
    document.addEventListener('keydown', handleKeyPress);
    
    // モーダルを閉じるボタンのイベントリスナー
    modal.querySelector('.modal-close')?.addEventListener('click', closeImageModal);
    modalContent.querySelector('.modal-close-btn')?.addEventListener('click', closeImageModal);
    
    // モーダル外をクリックで閉じる
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeImageModal();
        }
    });
}


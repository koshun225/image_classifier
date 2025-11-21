// 共通JavaScript

// モーダルの開閉
document.addEventListener('DOMContentLoaded', function() {
    // モーダルを開く
    const openModal = function(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'block';
        }
    };
    
    // モーダルを閉じる
    const closeModal = function(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'none';
        }
    };
    
    // 閉じるボタンのイベント
    document.querySelectorAll('.modal-close, .modal-close-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            const modal = this.closest('.modal');
            if (modal) {
                modal.style.display = 'none';
            }
        });
    });
    
    // モーダル外をクリックで閉じる
    document.querySelectorAll('.modal').forEach(function(modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === this) {
                this.style.display = 'none';
            }
        });
    });
});

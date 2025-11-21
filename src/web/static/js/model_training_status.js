/**
 * 学習実行・進捗画面のJavaScript
 */

let statusUpdateInterval = null;

// ステータス更新
async function updateStatus() {
    try {
        const response = await fetch(`/api/theme/${themeId}/training/status/${jobId}/`, {
            method: 'GET',
            headers: {
                'X-CSRFToken': csrfToken,
            },
        });
        
        const data = await response.json();
        
        if (data.success) {
            // ステータス更新
            const statusValue = document.getElementById('status-value');
            if (statusValue) {
                const statusLabels = {
                    'pending': '待機中',
                    'running': '実行中',
                    'completed': '完了',
                    'failed': '失敗',
                    'cancelled': 'キャンセル',
                };
                statusValue.textContent = statusLabels[data.status] || data.status;
                statusValue.className = `status-value status-${data.status}`;
            }
            
            // 開始時刻・完了時刻更新
            if (data.started_at) {
                const startedAt = document.getElementById('started-at');
                if (startedAt) {
                    startedAt.textContent = new Date(data.started_at).toLocaleString('ja-JP');
                }
            }
            
            if (data.completed_at) {
                const completedAt = document.getElementById('completed-at');
                if (completedAt) {
                    completedAt.textContent = new Date(data.completed_at).toLocaleString('ja-JP');
                }
            }
            
            // ログ更新
            const logContent = document.getElementById('log-content');
            if (logContent && data.log) {
                logContent.textContent = data.log;
                // 自動スクロール
                logContent.scrollTop = logContent.scrollHeight;
            }
            
            // 完了または失敗した場合は更新を停止
            if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
                if (statusUpdateInterval) {
                    clearInterval(statusUpdateInterval);
                    statusUpdateInterval = null;
                }
            }
        }
    } catch (error) {
        console.error('ステータス更新エラー:', error);
    }
}

// 初回読み込み時と定期的に更新
updateStatus();
statusUpdateInterval = setInterval(updateStatus, 3000); // 3秒ごとに更新

// ページを離れる際にクリーンアップ
window.addEventListener('beforeunload', () => {
    if (statusUpdateInterval) {
        clearInterval(statusUpdateInterval);
    }
});


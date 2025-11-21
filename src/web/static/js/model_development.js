/**
 * モデル開発画面のJavaScript
 */

const paramsDataScript = document.getElementById('params-data');
const optunaDataScript = document.getElementById('optuna-data');
const augumentsDataScript = document.getElementById('auguments-data');
const availableModelsScript = document.getElementById('available-models-data');
const registeredModelsScript = document.getElementById('registered-models-data');
const selectedModelIdScript = document.getElementById('selected-model-id-data');
const paramsState = paramsDataScript ? JSON.parse(paramsDataScript.textContent) : {};
const optunaState = optunaDataScript ? JSON.parse(optunaDataScript.textContent) : {};
const augumentsState = augumentsDataScript ? JSON.parse(augumentsDataScript.textContent) : {};
const availableModels = availableModelsScript ? JSON.parse(availableModelsScript.textContent) : [];
const registeredModels = registeredModelsScript ? JSON.parse(registeredModelsScript.textContent) : [];
const selectedModelId = selectedModelIdScript ? selectedModelIdScript.textContent.trim().replace(/^"|"$/g, '') : '';
const paramsFormEl = document.getElementById('params-form');
const augumentsFormEl = document.getElementById('auguments-form');
const valueTypeMap = {};
let selectedModelCheckpoint = null;  // 選択されたモデルのチェックポイントパス

const PARAM_META_KEYS = ['type', 'low', 'high', 'step', 'log', 'choices'];
const NON_TUNABLE_PATHS = new Set([
    'training.run_name',
    'training.num_epochs',
    'training.num_workers',
]);
let lrPreviewContext = null;

// タブ切り替え
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    });
});

function isParamNode(node) {
    if (!node || typeof node !== 'object' || Array.isArray(node)) return false;
    return Object.prototype.hasOwnProperty.call(node, 'value') ||
        Object.prototype.hasOwnProperty.call(node, 'type') ||
        Object.keys(node).some(key => PARAM_META_KEYS.includes(key));
}

function detectValueType(value) {
    if (typeof value === 'number') {
        return Number.isInteger(value) ? 'integer' : 'number';
    }
    if (typeof value === 'boolean') {
        return 'boolean';
    }
    if (value === null || value === undefined) {
        return 'string';
    }
    return 'string';
}

function buildValueTypeMap(node, path = []) {
    if (!node || typeof node !== 'object') return;
    Object.entries(node).forEach(([key, value]) => {
        const currentPath = [...path, key];
        if (isParamNode(value)) {
            const valueType = detectValueType(value.value);
            valueTypeMap[currentPath.join('.')] = valueType;
        } else if (typeof value === 'object') {
            buildValueTypeMap(value, currentPath);
        }
    });
}

function extractNodeValue(node) {
    if (node && typeof node === 'object' && !Array.isArray(node) && Object.prototype.hasOwnProperty.call(node, 'value')) {
        return node.value;
    }
    return node;
}

function getNodeByPath(path) {
    const keys = path.split('.');
    return keys.reduce((acc, key) => (acc ? acc[key] : undefined), paramsState);
}

function parseByType(raw, type) {
    if (raw === '' || raw === null || raw === undefined) {
        return null;
    }
    if (type === 'integer') {
        const parsed = parseInt(raw, 10);
        return Number.isNaN(parsed) ? raw : parsed;
    }
    if (type === 'number') {
        const parsed = parseFloat(raw);
        return Number.isNaN(parsed) ? raw : parsed;
    }
    if (type === 'boolean') {
        return raw === true || raw === 'true';
    }
    return raw;
}

function renderParamsForm() {
    if (!paramsFormEl) return;
    paramsFormEl.innerHTML = '';
    Object.keys(paramsState).forEach(section => {
        if (section === 'data' || section === 'optuna') {
            return;
        }
        const sectionEl = document.createElement('div');
        sectionEl.className = 'param-section';
        const header = document.createElement('h3');
        header.textContent = section;
        sectionEl.appendChild(header);
        renderSection(sectionEl, paramsState[section], [section]);
        paramsFormEl.appendChild(sectionEl);
    });
}

function renderSection(container, node, path) {
    if (!node || typeof node !== 'object') return;
    let entries = Object.entries(node);

    // trainingセクションは指定の順序で表示
    if (path.length === 1 && path[0] === 'training') {
        const preferredOrder = [
            'run_name',
            'num_epochs',
            'num_workers',
            'batch_size',
            'optimizer',
            'learning_rate',
            'scheduler'
        ];
        const orderedEntries = [];
        preferredOrder.forEach(key => {
            if (Object.prototype.hasOwnProperty.call(node, key)) {
                orderedEntries.push([key, node[key]]);
            }
        });
        const remainingEntries = entries.filter(([key]) => !preferredOrder.includes(key));
        entries = [...orderedEntries, ...remainingEntries];
    }

    entries.forEach(([key, value]) => {
        const currentPath = [...path, key];
        const fullPath = currentPath.join('.');
        
        // data.theme_idは非表示にする
        if (fullPath === 'data.theme_id') {
            return;
        }
        
        // schedulerとscheduler_paramsは特別な処理
        if (fullPath === 'training.scheduler' || fullPath === 'training.scheduler_params') {
            return; // 後で特別に処理する
        }
        
        if (isParamNode(value)) {
            container.appendChild(createParamRow(fullPath, key, value));
        } else if (typeof value === 'object') {
            const group = document.createElement('div');
            group.className = 'param-group';
            const title = document.createElement('div');
            title.className = 'param-group-title';
            title.textContent = key;
            group.appendChild(title);
            renderSection(group, value, currentPath);
            container.appendChild(group);
        }
    });
    
    // trainingセクションの場合は、scheduler設定を追加
    if (path.length === 1 && path[0] === 'training' && node.scheduler !== undefined) {
        container.appendChild(createSchedulerSection(node));
    }
}

function updateParamRowBackground(row, isTunable) {
    if (isTunable) {
        row.style.backgroundColor = '#e8f5e9';
        row.style.borderLeft = '3px solid #4caf50';
    } else {
        row.style.backgroundColor = '';
        row.style.borderLeft = '';
    }
}

function createParamRow(path, label, node) {
    const row = document.createElement('div');
    row.className = 'param-row';

    const isNonTunableField = NON_TUNABLE_PATHS.has(path);
    if (isNonTunableField) {
        delete node.type;
        PARAM_META_KEYS.forEach(key => {
            if (key !== 'type') {
                delete node[key];
            }
        });
    }

    // チューニング対象の場合、背景色を設定
    const isTunable = Boolean(node.type) && !node.readonly && !isNonTunableField;
    updateParamRowBackground(row, isTunable);

    const main = document.createElement('div');
    main.className = 'param-main';

    const info = document.createElement('div');
    info.className = 'param-info';
    info.innerHTML = `<div class="param-label">${label}</div><div class="param-path">${path}</div>`;

    const valueWrapper = document.createElement('div');
    valueWrapper.className = 'param-value';
    valueWrapper.appendChild(createValueInput(path, node));

    const actions = document.createElement('div');
    actions.className = 'param-actions';
    
    // readonlyフラグがある場合はチューニングトグルを表示しない
    if (!node.readonly && !isNonTunableField) {
        const tuneToggle = document.createElement('label');
        const toggle = document.createElement('input');
        toggle.type = 'checkbox';
        toggle.checked = Boolean(node.type);
        toggle.addEventListener('change', () => {
            if (toggle.checked) {
                // model.name、optimizer、batch_sizeの場合は常にcategorical
                if (path === 'model.name' || path === 'training.optimizer' || path === 'training.batch_size') {
                    node.type = 'categorical';
                    if (!Array.isArray(node.choices)) {
                        node.choices = [];
                    }
                } else {
                    node.type = node.type || 'float';
                    if (node.type === 'categorical' && !Array.isArray(node.choices)) {
                        node.choices = [];
                    }
                }
            } else {
                delete node.type;
                PARAM_META_KEYS.forEach(key => delete node[key]);
            }
            // 背景色を更新
            const isTunable = Boolean(node.type) && !node.readonly && !isNonTunableField;
            updateParamRowBackground(row, isTunable);
            renderTuningFields(tuningFields, node, path);
            // チューニング対象の数に応じてn_trialsを更新
            updateOptunaTrialsBasedOnTunableParams();
        });
        tuneToggle.appendChild(toggle);
        tuneToggle.appendChild(document.createTextNode('チューニングする'));
        actions.appendChild(tuneToggle);
    }

    main.appendChild(info);
    main.appendChild(valueWrapper);
    main.appendChild(actions);

    const tuningFields = document.createElement('div');
    tuningFields.className = 'tuning-fields';
    // readonlyフラグがある場合はチューニングフィールドを表示しない
    if (!node.readonly && !isNonTunableField) {
        renderTuningFields(tuningFields, node, path);
    }

    row.appendChild(main);
    row.appendChild(tuningFields);
    return row;
}

function createValueInput(path, node) {
    // readonlyフラグがある場合、またはdata.theme_idの場合は読み取り専用表示
    if (node.readonly || path === 'data.theme_id') {
        const input = document.createElement('input');
        input.type = 'text';
        input.value = node.value ?? '';
        input.readOnly = true;
        input.classList.add('form-control', 'readonly-input');
        input.style.backgroundColor = '#f5f5f5';
        input.style.cursor = 'not-allowed';
        return input;
    }
    
    // categoricalタイプの場合はselect要素を作成
    // node.typeが'categorical'の場合、またはchoicesが存在する場合
    const isCategorical = node.type === 'categorical';
    const hasChoices = node.choices && Array.isArray(node.choices) && node.choices.length > 0;
    
    // model.nameの場合は常にselect要素を作成
    if (path === 'model.name') {
        const select = document.createElement('select');
        select.className = 'form-control';
        // choicesが存在する場合はそれを使用、ない場合はavailableModelsを使用
        const choices = hasChoices ? node.choices : (typeof availableModels !== 'undefined' && Array.isArray(availableModels) ? availableModels : []);
        // node.choicesを更新（後で使用するため）
        if (!hasChoices && choices.length > 0) {
            node.choices = choices;
        }
        choices.forEach(choice => {
            const option = document.createElement('option');
            option.value = String(choice);
            option.textContent = String(choice);
            if (String(node.value) === String(choice)) {
                option.selected = true;
            }
            select.appendChild(option);
        });
        // デフォルト値が設定されていない場合は最初の選択肢を選択
        if (!select.value && choices.length > 0) {
            select.value = String(choices[0]);
            node.value = choices[0];
        }
        select.addEventListener('change', () => {
            const selectedValue = select.value;
            const choice = choices.find(c => String(c) === selectedValue);
            node.value = choice !== undefined ? choice : selectedValue;
        });
        return select;
    }
    
    // training.optimizerの場合は常にselect要素を作成
    if (path === 'training.optimizer') {
        const select = document.createElement('select');
        select.className = 'form-control';
        // choicesが存在する場合はそれを使用、ない場合はデフォルトの選択肢を使用
        const choices = hasChoices ? node.choices : ['Adam', 'SGD', 'AdamW'];
        // node.choicesを更新（後で使用するため）
        if (!hasChoices) {
            node.choices = choices;
        }
        choices.forEach(choice => {
            const option = document.createElement('option');
            option.value = String(choice);
            option.textContent = String(choice);
            if (String(node.value) === String(choice)) {
                option.selected = true;
            }
            select.appendChild(option);
        });
        // デフォルト値が設定されていない場合は最初の選択肢を選択
        if (!select.value && choices.length > 0) {
            select.value = String(choices[0]);
            node.value = choices[0];
        }
        select.addEventListener('change', () => {
            const selectedValue = select.value;
            const choice = choices.find(c => String(c) === selectedValue);
            node.value = choice !== undefined ? choice : selectedValue;
        });
        return select;
    }
    
    // choicesが存在する場合はselect要素を作成（categoricalタイプであることを確認）
    if (hasChoices && isCategorical) {
        const select = document.createElement('select');
        select.className = 'form-control';
        node.choices.forEach(choice => {
            const option = document.createElement('option');
            option.value = String(choice);
            option.textContent = String(choice);
            if (String(node.value) === String(choice)) {
                option.selected = true;
            }
            select.appendChild(option);
        });
        // デフォルト値が設定されていない場合は最初の選択肢を選択
        if (!select.value && node.choices.length > 0) {
            select.value = String(node.choices[0]);
            node.value = node.choices[0];
        }
        select.addEventListener('change', () => {
            // 数値の場合は数値に変換、文字列の場合は文字列のまま
            const selectedValue = select.value;
            const choice = node.choices.find(c => String(c) === selectedValue);
            node.value = choice !== undefined ? choice : selectedValue;
        });
        return select;
    }
    
    const type = valueTypeMap[path] || detectValueType(node.value);
    let input;
    if (type === 'boolean') {
        input = document.createElement('select');
        input.innerHTML = `
            <option value="true">true</option>
            <option value="false">false</option>
        `;
        input.value = node.value ? 'true' : 'false';
        input.addEventListener('change', () => {
            node.value = input.value === 'true';
        });
    } else {
        input = document.createElement('input');
        input.type = (type === 'number' || type === 'integer') ? 'number' : 'text';
        if (type === 'number') input.step = 'any';
        if (type === 'integer') input.step = '1';
        input.value = node.value ?? '';
        input.placeholder = '値を入力';
        input.addEventListener('input', () => {
            node.value = parseByType(input.value, type);
            if (path === 'training.learning_rate' || path === 'training.num_epochs') {
                triggerLearningRatePreview();
            }
        });
    }
    input.classList.add('form-control');
    return input;
}

function renderTuningFields(container, node, path) {
    container.innerHTML = '';
    if (!node.type) {
        container.classList.remove('active');
        return;
    }
    container.classList.add('active');

    const grid = document.createElement('div');
    grid.className = 'tuning-grid';

    // model.name、optimizer、batch_sizeの場合は種類選択を表示しない（常にcategorical）
    if (path !== 'model.name' && path !== 'training.optimizer' && path !== 'training.batch_size') {
        const typeSelect = document.createElement('select');
        typeSelect.className = 'form-control';
        ['float', 'int', 'categorical'].forEach(option => {
            const opt = document.createElement('option');
            opt.value = option;
            opt.textContent = option;
            typeSelect.appendChild(opt);
        });
        typeSelect.value = node.type;
        typeSelect.addEventListener('change', () => {
            node.type = typeSelect.value;
            PARAM_META_KEYS.forEach(key => delete node[key]);
            if (node.type === 'categorical') {
                node.choices = [];
            }
            renderTuningFields(container, node, path);
        });

        const typeField = document.createElement('div');
        typeField.className = 'form-group';
        typeField.innerHTML = '<label>種類</label>';
        typeField.appendChild(typeSelect);
        grid.appendChild(typeField);
    } else {
        // model.name、optimizer、batch_sizeの場合は常にcategoricalとして扱う
        node.type = 'categorical';
    }

    if (node.type === 'float' || node.type === 'int') {
        const lowField = document.createElement('div');
        lowField.className = 'form-group';
        lowField.innerHTML = '<label>最小値 (low)</label>';
        const lowInput = document.createElement('input');
        lowInput.type = 'number';
        lowInput.step = node.type === 'int' ? '1' : 'any';
        lowInput.className = 'form-control';
        lowInput.value = node.low ?? '';
        lowInput.addEventListener('input', () => {
            node.low = lowInput.value === '' ? null : parseFloat(lowInput.value);
        });
        lowField.appendChild(lowInput);
        grid.appendChild(lowField);

        const highField = document.createElement('div');
        highField.className = 'form-group';
        highField.innerHTML = '<label>最大値 (high)</label>';
        const highInput = document.createElement('input');
        highInput.type = 'number';
        highInput.step = node.type === 'int' ? '1' : 'any';
        highInput.className = 'form-control';
        highInput.value = node.high ?? '';
        highInput.addEventListener('input', () => {
            node.high = highInput.value === '' ? null : parseFloat(highInput.value);
        });
        highField.appendChild(highInput);
        grid.appendChild(highField);
    }

    if (node.type === 'int') {
        const stepField = document.createElement('div');
        stepField.className = 'form-group';
        stepField.innerHTML = '<label>刻み幅 (step)</label>';
        const stepInput = document.createElement('input');
        stepInput.type = 'number';
        stepInput.step = '1';
        stepInput.className = 'form-control';
        stepInput.value = node.step ?? '';
        stepInput.addEventListener('input', () => {
            node.step = stepInput.value === '' ? null : parseInt(stepInput.value, 10);
        });
        stepField.appendChild(stepInput);
        grid.appendChild(stepField);
    }

    if (node.type === 'float') {
        const logField = document.createElement('div');
        logField.className = 'form-group';
        const label = document.createElement('label');
        label.textContent = '対数スケール (log)';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = Boolean(node.log);
        checkbox.addEventListener('change', () => {
            node.log = checkbox.checked;
        });
        logField.appendChild(label);
        logField.appendChild(checkbox);
        grid.appendChild(logField);
    }

    if (node.type === 'categorical') {
        // model.nameの場合はチェックボックスで複数選択
        if (path === 'model.name' && typeof availableModels !== 'undefined' && Array.isArray(availableModels)) {
            const choicesField = document.createElement('div');
            choicesField.className = 'form-group';
            choicesField.style.gridColumn = '1 / -1'; // 全幅を使用
            
            const label = document.createElement('label');
            label.textContent = 'チューニング対象のモデル（複数選択可）';
            choicesField.appendChild(label);
            
            const checkboxContainer = document.createElement('div');
            checkboxContainer.className = 'checkbox-group';
            checkboxContainer.style.display = 'flex';
            checkboxContainer.style.flexDirection = 'column';
            checkboxContainer.style.gap = '0.5rem';
            checkboxContainer.style.marginTop = '0.5rem';
            
            // 表示する選択肢は常にavailableModelsを使用
            const allChoices = availableModels;
            
            // 現在選択されている値を配列として取得
            // node.choicesが配列の場合はそれを使用、そうでない場合はnode.valueから配列を作成
            let currentChoices = [];
            if (Array.isArray(node.choices) && node.choices.length > 0) {
                currentChoices = [...node.choices];
            } else if (node.value) {
                currentChoices = [node.value];
            }
            
            allChoices.forEach((choice, index) => {
                const checkboxWrapper = document.createElement('div');
                checkboxWrapper.className = 'checkbox-option';
                checkboxWrapper.style.display = 'flex';
                checkboxWrapper.style.alignItems = 'center';
                checkboxWrapper.style.gap = '0.5rem';
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `tuning-model-${path}-${index}`;
                checkbox.value = String(choice);
                checkbox.checked = currentChoices.some(c => String(c) === String(choice));
                
                checkbox.addEventListener('change', () => {
                    // 選択された値を配列として更新
                    if (checkbox.checked) {
                        if (!currentChoices.some(c => String(c) === String(choice))) {
                            currentChoices.push(choice);
                        }
                    } else {
                        const idx = currentChoices.findIndex(c => String(c) === String(choice));
                        if (idx !== -1) {
                            currentChoices.splice(idx, 1);
                        }
                    }
                    // node.choicesに選択された値を保存
                    node.choices = [...currentChoices];
                    // node.valueは最初の選択値を設定（互換性のため）
                    if (currentChoices.length > 0) {
                        node.value = currentChoices[0];
                    } else {
                        node.value = null;
                    }
                });
                
                const checkboxLabel = document.createElement('label');
                checkboxLabel.htmlFor = `tuning-model-${path}-${index}`;
                checkboxLabel.textContent = String(choice);
                checkboxLabel.style.margin = '0';
                checkboxLabel.style.cursor = 'pointer';
                
                checkboxWrapper.appendChild(checkbox);
                checkboxWrapper.appendChild(checkboxLabel);
                checkboxContainer.appendChild(checkboxWrapper);
            });
            
            // choicesを更新（availableModelsを使用する場合）
            if (!node.choices || node.choices.length === 0) {
                node.choices = [];
            }
            
            choicesField.appendChild(checkboxContainer);
            grid.appendChild(choicesField);
        } else if (path === 'training.optimizer') {
            // optimizerの場合はチェックボックスで複数選択
            const optimizerChoices = ['Adam', 'SGD', 'AdamW'];
            
            const choicesField = document.createElement('div');
            choicesField.className = 'form-group';
            choicesField.style.gridColumn = '1 / -1'; // 全幅を使用
            
            const label = document.createElement('label');
            label.textContent = 'チューニング対象のオプティマイザー（複数選択可）';
            choicesField.appendChild(label);
            
            const checkboxContainer = document.createElement('div');
            checkboxContainer.className = 'checkbox-group';
            checkboxContainer.style.display = 'flex';
            checkboxContainer.style.flexDirection = 'column';
            checkboxContainer.style.gap = '0.5rem';
            checkboxContainer.style.marginTop = '0.5rem';
            
            // 現在選択されている値を配列として取得
            let currentChoices = [];
            if (Array.isArray(node.choices) && node.choices.length > 0) {
                currentChoices = [...node.choices];
            } else if (node.value) {
                currentChoices = [node.value];
            }
            
            optimizerChoices.forEach((choice, index) => {
                const checkboxWrapper = document.createElement('div');
                checkboxWrapper.className = 'checkbox-option';
                checkboxWrapper.style.display = 'flex';
                checkboxWrapper.style.alignItems = 'center';
                checkboxWrapper.style.gap = '0.5rem';
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `tuning-optimizer-${path}-${index}`;
                checkbox.value = String(choice);
                checkbox.checked = currentChoices.some(c => String(c) === String(choice));
                
                checkbox.addEventListener('change', () => {
                    // 選択された値を配列として更新
                    if (checkbox.checked) {
                        if (!currentChoices.some(c => String(c) === String(choice))) {
                            currentChoices.push(choice);
                        }
                    } else {
                        const idx = currentChoices.findIndex(c => String(c) === String(choice));
                        if (idx !== -1) {
                            currentChoices.splice(idx, 1);
                        }
                    }
                    // node.choicesに選択された値を保存
                    node.choices = [...currentChoices];
                    // node.valueは最初の選択値を設定（互換性のため）
                    if (currentChoices.length > 0) {
                        node.value = currentChoices[0];
                    } else {
                        node.value = null;
                    }
                });
                
                const checkboxLabel = document.createElement('label');
                checkboxLabel.htmlFor = `tuning-optimizer-${path}-${index}`;
                checkboxLabel.textContent = String(choice);
                checkboxLabel.style.margin = '0';
                checkboxLabel.style.cursor = 'pointer';
                
                checkboxWrapper.appendChild(checkbox);
                checkboxWrapper.appendChild(checkboxLabel);
                checkboxContainer.appendChild(checkboxWrapper);
            });
            
            // choicesを更新
            if (!node.choices || node.choices.length === 0) {
                node.choices = [];
            }
            
            choicesField.appendChild(checkboxContainer);
            grid.appendChild(choicesField);
        } else if (path === 'training.batch_size') {
            // batch_sizeの場合はチェックボックスで複数選択
            const batchSizeChoices = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
            
            const choicesField = document.createElement('div');
            choicesField.className = 'form-group';
            choicesField.style.gridColumn = '1 / -1'; // 全幅を使用
            
            const label = document.createElement('label');
            label.textContent = 'チューニング対象のバッチサイズ（複数選択可）';
            choicesField.appendChild(label);
            
            const checkboxContainer = document.createElement('div');
            checkboxContainer.className = 'checkbox-group';
            checkboxContainer.style.display = 'flex';
            checkboxContainer.style.flexDirection = 'column';
            checkboxContainer.style.gap = '0.5rem';
            checkboxContainer.style.marginTop = '0.5rem';
            
            // 現在選択されている値を配列として取得
            let currentChoices = [];
            if (Array.isArray(node.choices) && node.choices.length > 0) {
                currentChoices = [...node.choices];
            } else if (node.value !== null && node.value !== undefined) {
                currentChoices = [node.value];
            }
            
            batchSizeChoices.forEach((choice, index) => {
                const checkboxWrapper = document.createElement('div');
                checkboxWrapper.className = 'checkbox-option';
                checkboxWrapper.style.display = 'flex';
                checkboxWrapper.style.alignItems = 'center';
                checkboxWrapper.style.gap = '0.5rem';
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `tuning-batch-size-${path}-${index}`;
                checkbox.value = String(choice);
                checkbox.checked = currentChoices.some(c => Number(c) === Number(choice));
                
                checkbox.addEventListener('change', () => {
                    // 選択された値を配列として更新
                    if (checkbox.checked) {
                        if (!currentChoices.some(c => Number(c) === Number(choice))) {
                            currentChoices.push(choice);
                        }
                    } else {
                        const idx = currentChoices.findIndex(c => Number(c) === Number(choice));
                        if (idx !== -1) {
                            currentChoices.splice(idx, 1);
                        }
                    }
                    // node.choicesに選択された値を保存
                    node.choices = [...currentChoices];
                    // node.valueは最初の選択値を設定（互換性のため）
                    if (currentChoices.length > 0) {
                        node.value = currentChoices[0];
                    } else {
                        node.value = null;
                    }
                });
                
                const checkboxLabel = document.createElement('label');
                checkboxLabel.htmlFor = `tuning-batch-size-${path}-${index}`;
                checkboxLabel.textContent = String(choice);
                checkboxLabel.style.margin = '0';
                checkboxLabel.style.cursor = 'pointer';
                
                checkboxWrapper.appendChild(checkbox);
                checkboxWrapper.appendChild(checkboxLabel);
                checkboxContainer.appendChild(checkboxWrapper);
            });
            
            // choicesを更新
            if (!node.choices || node.choices.length === 0) {
                node.choices = [];
            }
            
            choicesField.appendChild(checkboxContainer);
            grid.appendChild(choicesField);
        } else {
            // その他のcategoricalタイプは従来通りテキストエリア
            const choicesField = document.createElement('div');
            choicesField.className = 'form-group';
            choicesField.innerHTML = '<label>候補値 (1行に1つ)</label>';
            const textarea = document.createElement('textarea');
            textarea.className = 'form-control';
            textarea.rows = 3;
            textarea.value = (node.choices || []).join('\n');
            textarea.addEventListener('input', () => {
                node.choices = textarea.value
                    .split('\n')
                    .map(item => item.trim())
                    .filter(Boolean);
            });
            choicesField.appendChild(textarea);
            grid.appendChild(choicesField);
        }
    }

    container.appendChild(grid);
}

function createSchedulerSection(trainingNode) {
    const section = document.createElement('div');
    section.className = 'scheduler-section';
    section.style.marginTop = '1.5rem';
    section.style.padding = '1rem';
    section.style.border = '1px solid #e0e0e0';
    section.style.borderRadius = '8px';
    section.style.backgroundColor = '#fafbfc';
    
    const header = document.createElement('h4');
    header.textContent = '学習率スケジューラ';
    header.style.marginTop = '0';
    header.style.marginBottom = '1rem';
    section.appendChild(header);
    
    // スケジューラの種類選択
    const schedulerTypeDiv = document.createElement('div');
    schedulerTypeDiv.className = 'form-group';
    const schedulerLabel = document.createElement('label');
    schedulerLabel.textContent = 'スケジューラの種類';
    schedulerTypeDiv.appendChild(schedulerLabel);
    
    const schedulerSelect = document.createElement('select');
    schedulerSelect.className = 'form-control';
    schedulerSelect.id = 'scheduler-type';
    schedulerSelect.innerHTML = `
        <option value="">なし (None)</option>
        <option value="StepLR">StepLR</option>
        <option value="CosineAnnealingLR">CosineAnnealingLR</option>
        <option value="ReduceLROnPlateau">ReduceLROnPlateau</option>
    `;
    
    // 現在の値を設定
    const currentScheduler = trainingNode.scheduler?.value || trainingNode.scheduler || null;
    schedulerSelect.value = currentScheduler || '';
    
    schedulerSelect.addEventListener('change', () => {
        const schedulerValue = schedulerSelect.value || null;
        if (!trainingNode.scheduler) {
            trainingNode.scheduler = {};
        }
        if (typeof trainingNode.scheduler === 'object' && 'value' in trainingNode.scheduler) {
            trainingNode.scheduler.value = schedulerValue;
        } else {
            trainingNode.scheduler = { value: schedulerValue };
        }
        if (schedulerValue === 'StepLR') {
            ensureStepLRDefaults(trainingNode);
        }
        renderSchedulerParams(schedulerParamsDiv, schedulerValue, trainingNode);
    });
    
    schedulerTypeDiv.appendChild(schedulerSelect);
    section.appendChild(schedulerTypeDiv);
    
    // スケジューラパラメータのコンテナ
    const schedulerParamsDiv = document.createElement('div');
    schedulerParamsDiv.id = 'scheduler-params';
    schedulerParamsDiv.className = 'scheduler-params';
    section.appendChild(schedulerParamsDiv);
    
    // 初期表示
    renderSchedulerParams(schedulerParamsDiv, currentScheduler, trainingNode);

    const lrPreview = document.createElement('div');
    lrPreview.className = 'lr-preview';
    lrPreview.style.marginTop = '1.5rem';

    const lrHeader = document.createElement('h5');
    lrHeader.textContent = 'Learning Rate推移';
    lrHeader.style.margin = '0 0 0.5rem 0';
    lrPreview.appendChild(lrHeader);

    const lrCanvas = document.createElement('canvas');
    lrCanvas.width = 480;
    lrCanvas.height = 220;
    lrCanvas.className = 'lr-preview-canvas';
    lrPreview.appendChild(lrCanvas);

    const lrInfo = document.createElement('div');
    lrInfo.className = 'lr-preview-info';
    lrInfo.style.fontSize = '0.9rem';
    lrInfo.style.marginTop = '0.5rem';
    lrPreview.appendChild(lrInfo);

    const lrNote = document.createElement('p');
    lrNote.className = 'lr-preview-note';
    lrNote.style.fontSize = '0.85rem';
    lrNote.style.color = '#666';
    lrNote.style.marginTop = '0.25rem';
    lrPreview.appendChild(lrNote);

    section.appendChild(lrPreview);

    lrPreviewContext = {
        canvas: lrCanvas,
        trainingNode,
        infoEl: lrInfo,
        noteEl: lrNote,
    };
    renderLearningRatePreview();
    
    return section;
}

function renderSchedulerParams(container, schedulerType, trainingNode) {
    container.innerHTML = '';
    
    if (!schedulerType || schedulerType === '') {
        triggerLearningRatePreview();
        return;
    }
    
    // scheduler_paramsを初期化
    if (!trainingNode.scheduler_params) {
        trainingNode.scheduler_params = {};
    }
    if (typeof trainingNode.scheduler_params !== 'object') {
        trainingNode.scheduler_params = {};
    }
    
    const paramsDiv = document.createElement('div');
    paramsDiv.className = 'scheduler-params-grid';
    paramsDiv.style.display = 'grid';
    paramsDiv.style.gridTemplateColumns = 'repeat(auto-fit, minmax(200px, 1fr))';
    paramsDiv.style.gap = '1rem';
    paramsDiv.style.marginTop = '1rem';
    
    if (schedulerType === 'StepLR') {
        const currentStepSize = ensureStepLRDefaults(trainingNode);
        const stepSizeDiv = document.createElement('div');
        stepSizeDiv.className = 'form-group';
        const stepSizeLabel = document.createElement('label');
        stepSizeLabel.textContent = 'step_size';
        const stepSizeInput = document.createElement('input');
        stepSizeInput.type = 'number';
        stepSizeInput.className = 'form-control';
        stepSizeInput.min = '1';
        stepSizeInput.step = '1';
        stepSizeInput.value = currentStepSize;
        stepSizeInput.addEventListener('input', () => {
            const parsed = parseInt(stepSizeInput.value, 10);
            if (Number.isFinite(parsed) && parsed > 0) {
                trainingNode.scheduler_params.step_size = parsed;
            } else if (stepSizeInput.value === '') {
                delete trainingNode.scheduler_params.step_size;
            }
            triggerLearningRatePreview();
        });
        stepSizeDiv.appendChild(stepSizeLabel);
        stepSizeDiv.appendChild(stepSizeInput);
        paramsDiv.appendChild(stepSizeDiv);
        
        const gammaDiv = document.createElement('div');
        gammaDiv.className = 'form-group';
        const gammaLabel = document.createElement('label');
        gammaLabel.textContent = 'gamma';
        const gammaInput = document.createElement('input');
        gammaInput.type = 'number';
        gammaInput.className = 'form-control';
        gammaInput.min = '0';
        gammaInput.max = '1';
        gammaInput.step = '0.01';
        gammaInput.value = trainingNode.scheduler_params.gamma || 0.1;
        gammaInput.addEventListener('input', () => {
            trainingNode.scheduler_params.gamma = parseFloat(gammaInput.value);
            triggerLearningRatePreview();
        });
        gammaDiv.appendChild(gammaLabel);
        gammaDiv.appendChild(gammaInput);
        paramsDiv.appendChild(gammaDiv);
        
    } else if (schedulerType === 'CosineAnnealingLR') {
        const tMaxDiv = document.createElement('div');
        tMaxDiv.className = 'form-group';
        const tMaxLabel = document.createElement('label');
        tMaxLabel.textContent = 'T_max';
        const tMaxInput = document.createElement('input');
        tMaxInput.type = 'number';
        tMaxInput.className = 'form-control';
        tMaxInput.min = '1';
        tMaxInput.step = '1';
        tMaxInput.value = trainingNode.scheduler_params.T_max || 100;
        tMaxInput.addEventListener('input', () => {
            trainingNode.scheduler_params.T_max = parseInt(tMaxInput.value, 10);
            triggerLearningRatePreview();
        });
        tMaxDiv.appendChild(tMaxLabel);
        tMaxDiv.appendChild(tMaxInput);
        paramsDiv.appendChild(tMaxDiv);
        
        const etaMinDiv = document.createElement('div');
        etaMinDiv.className = 'form-group';
        const etaMinLabel = document.createElement('label');
        etaMinLabel.textContent = 'eta_min';
        const etaMinInput = document.createElement('input');
        etaMinInput.type = 'number';
        etaMinInput.className = 'form-control';
        etaMinInput.min = '0';
        etaMinInput.step = '0.0001';
        etaMinInput.value = trainingNode.scheduler_params.eta_min || 0;
        etaMinInput.addEventListener('input', () => {
            trainingNode.scheduler_params.eta_min = parseFloat(etaMinInput.value);
            triggerLearningRatePreview();
        });
        etaMinDiv.appendChild(etaMinLabel);
        etaMinDiv.appendChild(etaMinInput);
        paramsDiv.appendChild(etaMinDiv);
        
    } else if (schedulerType === 'ReduceLROnPlateau') {
        const modeDiv = document.createElement('div');
        modeDiv.className = 'form-group';
        const modeLabel = document.createElement('label');
        modeLabel.textContent = 'mode';
        const modeSelect = document.createElement('select');
        modeSelect.className = 'form-control';
        modeSelect.innerHTML = `
            <option value="min">min</option>
            <option value="max">max</option>
        `;
        modeSelect.value = trainingNode.scheduler_params.mode || 'min';
        modeSelect.addEventListener('change', () => {
            trainingNode.scheduler_params.mode = modeSelect.value;
            triggerLearningRatePreview();
        });
        modeDiv.appendChild(modeLabel);
        modeDiv.appendChild(modeSelect);
        paramsDiv.appendChild(modeDiv);
        
        const factorDiv = document.createElement('div');
        factorDiv.className = 'form-group';
        const factorLabel = document.createElement('label');
        factorLabel.textContent = 'factor';
        const factorInput = document.createElement('input');
        factorInput.type = 'number';
        factorInput.className = 'form-control';
        factorInput.min = '0';
        factorInput.max = '1';
        factorInput.step = '0.01';
        factorInput.value = trainingNode.scheduler_params.factor || 0.1;
        factorInput.addEventListener('input', () => {
            trainingNode.scheduler_params.factor = parseFloat(factorInput.value);
            triggerLearningRatePreview();
        });
        factorDiv.appendChild(factorLabel);
        factorDiv.appendChild(factorInput);
        paramsDiv.appendChild(factorDiv);
        
        const patienceDiv = document.createElement('div');
        patienceDiv.className = 'form-group';
        const patienceLabel = document.createElement('label');
        patienceLabel.textContent = 'patience';
        const patienceInput = document.createElement('input');
        patienceInput.type = 'number';
        patienceInput.className = 'form-control';
        patienceInput.min = '1';
        patienceInput.step = '1';
        patienceInput.value = trainingNode.scheduler_params.patience || 10;
        patienceInput.addEventListener('input', () => {
            trainingNode.scheduler_params.patience = parseInt(patienceInput.value, 10);
            triggerLearningRatePreview();
        });
        patienceDiv.appendChild(patienceLabel);
        patienceDiv.appendChild(patienceInput);
        paramsDiv.appendChild(patienceDiv);
        
        const monitorDiv = document.createElement('div');
        monitorDiv.className = 'form-group';
        const monitorLabel = document.createElement('label');
        monitorLabel.textContent = 'monitor';
        const monitorInput = document.createElement('input');
        monitorInput.type = 'text';
        monitorInput.className = 'form-control';
        monitorInput.value = trainingNode.scheduler_params.monitor || 'val_loss';
        monitorInput.addEventListener('input', () => {
            trainingNode.scheduler_params.monitor = monitorInput.value;
            triggerLearningRatePreview();
        });
        monitorDiv.appendChild(monitorLabel);
        monitorDiv.appendChild(monitorInput);
        paramsDiv.appendChild(monitorDiv);
    }
    
    container.appendChild(paramsDiv);
    triggerLearningRatePreview();
}

function triggerLearningRatePreview() {
    if (!lrPreviewContext) return;
    window.requestAnimationFrame(renderLearningRatePreview);
}

function renderLearningRatePreview() {
    if (!lrPreviewContext) return;
    const { canvas, trainingNode, infoEl, noteEl } = lrPreviewContext;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const series = computeLearningRateSeries(trainingNode);
    if (!series.length) {
        if (infoEl) infoEl.textContent = '学習率情報が不足しています。';
        if (noteEl) noteEl.textContent = '';
        return;
    }
    
    const values = series.map(item => item.value);
    const maxVal = Math.max(...values);
    const minVal = Math.min(...values);
    const padding = 32;
    const width = canvas.width - padding * 2;
    const height = canvas.height - padding * 2;
    const valueRange = maxVal - minVal || 1;
    
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, padding + height);
    ctx.lineTo(padding + width, padding + height);
    ctx.stroke();
    
    ctx.strokeStyle = '#1a73e8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((point, idx) => {
        const x = padding + (width * (idx / Math.max(series.length - 1, 1)));
        const yNorm = (point.value - minVal) / valueRange;
        const y = padding + height - yNorm * height;
        if (idx === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
    
    ctx.fillStyle = '#1a73e8';
    series.forEach((point, idx) => {
        const x = padding + (width * (idx / Math.max(series.length - 1, 1)));
        const yNorm = (point.value - minVal) / valueRange;
        const y = padding + height - yNorm * height;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
    });
    
    if (infoEl) {
        const first = series[0];
        const last = series[series.length - 1];
        infoEl.textContent = `初期: ${first.value.toPrecision(4)} / 最終: ${last.value.toPrecision(4)}`;
    }
    
    if (noteEl) {
        const schedulerType = (trainingNode.scheduler && trainingNode.scheduler.value) || null;
        if (schedulerType === 'ReduceLROnPlateau') {
            noteEl.textContent = '※ ReduceLROnPlateauは指標に応じて変化するため、ここでは概形を表示しています。';
        } else if (!schedulerType) {
            noteEl.textContent = '※ スケジューラ未設定時は一定の学習率になります。';
        } else {
            noteEl.textContent = '';
        }
    }
}

function computeLearningRateSeries(trainingNode) {
    const baseLr = Number(extractNodeValue(trainingNode.learning_rate));
    const totalEpochs = Number(extractNodeValue(trainingNode.num_epochs) ?? 0);
    if (!Number.isFinite(baseLr) || baseLr <= 0 || totalEpochs <= 0) {
        return [];
    }
    const schedulerType = (trainingNode.scheduler && trainingNode.scheduler.value) || null;
    const params = trainingNode.scheduler_params || {};
    const series = [];
    
    for (let epoch = 0; epoch <= totalEpochs; epoch++) {
        let lr = baseLr;
        if (schedulerType === 'StepLR') {
            const stepSize = Math.max(1, parseInt(params.step_size ?? 10, 10));
            const gamma = Number(params.gamma ?? 0.1);
            const steps = Math.floor(epoch / stepSize);
            lr = baseLr * Math.pow(gamma, steps);
        } else if (schedulerType === 'CosineAnnealingLR') {
            const tMaxValue = params.T_max ?? (totalEpochs || 1);
            const tMax = Math.max(1, parseInt(tMaxValue, 10));
            const etaMin = Number(params.eta_min ?? 0);
            const progress = Math.min(epoch, tMax);
            lr = etaMin + 0.5 * (baseLr - etaMin) * (1 + Math.cos(Math.PI * progress / tMax));
        } else if (schedulerType === 'ReduceLROnPlateau') {
            const patience = Math.max(1, parseInt(params.patience ?? 10, 10));
            const factor = Number(params.factor ?? 0.1);
            const reductions = Math.floor(epoch / patience);
            lr = baseLr * Math.pow(Math.max(Math.min(factor, 0.999), 1e-6), reductions);
        }
        series.push({
            epoch,
            value: Number.isFinite(lr) ? lr : 0,
        });
    }
    return series;
}

function ensureStepLRDefaults(trainingNode) {
    if (!trainingNode.scheduler_params || typeof trainingNode.scheduler_params !== 'object') {
        trainingNode.scheduler_params = {};
    }
    let stepSize = parseInt(trainingNode.scheduler_params.step_size, 10);
    if (!Number.isFinite(stepSize) || stepSize <= 0) {
        stepSize = 10;
    }
    trainingNode.scheduler_params.step_size = stepSize;
    return stepSize;
}

function sanitizeParamsState(node) {
    if (isParamNode(node)) {
        const cleaned = {};
        Object.entries(node).forEach(([key, value]) => {
            if (value === undefined) return;
            if (key === 'choices' && Array.isArray(value)) {
                cleaned[key] = value.filter(item => item !== '');
            } else {
                cleaned[key] = value;
            }
        });
        return cleaned;
    }
    if (typeof node === 'object' && node !== null) {
        const result = {};
        Object.entries(node).forEach(([key, value]) => {
            result[key] = sanitizeParamsState(value);
        });
        return result;
    }
    return node;
}

function initOptunaForm() {
    const metricInput = document.getElementById('optuna-metric');
    const directionSelect = document.getElementById('optuna-direction');
    const nTrialsInput = document.getElementById('optuna-ntrials');
    const timeoutInput = document.getElementById('optuna-timeout');

    optunaState.direction = optunaState.direction || 'maximize';

    if (metricInput) {
        metricInput.value = optunaState.metric || '';
        metricInput.addEventListener('input', () => {
            optunaState.metric = metricInput.value;
        });
    }
    if (directionSelect) {
        directionSelect.value = optunaState.direction;
        directionSelect.addEventListener('change', () => {
            optunaState.direction = directionSelect.value;
        });
    }
    if (nTrialsInput) {
        // 警告メッセージ用の要素を作成
        const warningDiv = document.createElement('div');
        warningDiv.id = 'optuna-ntrials-warning';
        warningDiv.className = 'optuna-warning';
        warningDiv.style.display = 'none';
        warningDiv.style.color = '#d32f2f';
        warningDiv.style.fontSize = '0.875rem';
        warningDiv.style.marginTop = '0.25rem';
        warningDiv.textContent = 'チューニングするにチェックを入れてください';
        
        // 試行回数の入力フィールドの親要素に警告メッセージを追加
        const nTrialsFormGroup = nTrialsInput.closest('.form-group');
        if (nTrialsFormGroup) {
            nTrialsFormGroup.appendChild(warningDiv);
        }
        
        // 初期化時にチューニング対象の数に応じてn_trialsを設定
        updateOptunaTrialsBasedOnTunableParams();
        // updateOptunaTrialsBasedOnTunableParams内で既にvalueを設定しているが、念のため再度設定
        nTrialsInput.value = optunaState.n_trials ?? '';
        
        // フォーカス時（編集開始時）に警告をチェック
        nTrialsInput.addEventListener('focus', () => {
            const tunableCount = countTunableParams(paramsState);
            if (tunableCount === 0) {
                warningDiv.style.display = 'block';
            }
        });
        
        // 入力時に警告をチェック
        nTrialsInput.addEventListener('input', () => {
            const newValue = nTrialsInput.value === '' ? null : parseInt(nTrialsInput.value, 10);
            const tunableCount = countTunableParams(paramsState);
            
            if (tunableCount === 0) {
                if (newValue !== 1 && newValue !== null) {
                    // 1以外の値を入力しようとした場合、警告を表示
                    warningDiv.style.display = 'block';
                    optunaState.n_trials = 1;
                    nTrialsInput.value = '1';
                } else {
                    // 1が入力された場合、警告を非表示
                    warningDiv.style.display = 'none';
                    optunaState.n_trials = 1;
                }
            } else {
                // チューニング対象がある場合、警告を非表示
                warningDiv.style.display = 'none';
                optunaState.n_trials = newValue;
            }
        });
        
        // フォーカスアウト時（編集終了時）に警告を非表示
        nTrialsInput.addEventListener('blur', () => {
            const tunableCount = countTunableParams(paramsState);
            if (tunableCount === 0) {
                // チューニング対象が0件の場合、値を1に戻す
                optunaState.n_trials = 1;
                nTrialsInput.value = '1';
                warningDiv.style.display = 'none';
            }
        });
    }
    if (timeoutInput) {
        timeoutInput.value = optunaState.timeout ?? '';
        timeoutInput.addEventListener('input', () => {
            optunaState.timeout = timeoutInput.value === '' ? null : parseInt(timeoutInput.value, 10);
        });
    }
}

function countTunableParams(node, path = []) {
    let count = 0;
    if (!node || typeof node !== 'object' || Array.isArray(node)) return count;
    
    Object.entries(node).forEach(([key, value]) => {
        const currentPath = [...path, key];
        const fullPath = currentPath.join('.');
        
        // 非チューニング対象やdataセクション、optunaセクションはスキップ
        if (NON_TUNABLE_PATHS.has(fullPath) || fullPath.startsWith('data.') || fullPath.startsWith('optuna.')) {
            return;
        }
        
        if (isParamNode(value)) {
            // typeプロパティが存在する場合はチューニング対象
            if (value.type) {
                count++;
            }
        } else if (typeof value === 'object' && value !== null) {
            count += countTunableParams(value, currentPath);
        }
    });
    
    return count;
}

function updateOptunaTrialsBasedOnTunableParams() {
    const tunableCount = countTunableParams(paramsState);
    const nTrialsInput = document.getElementById('optuna-ntrials');
    const warningDiv = document.getElementById('optuna-ntrials-warning');
    
    if (tunableCount === 0) {
        // チューニング対象が0の場合、n_trialsを1に設定
        optunaState.n_trials = 1;
        if (nTrialsInput) {
            nTrialsInput.value = '1';
            // 入力フィールドの値を強制的に更新（表示を確実に反映）
            nTrialsInput.dispatchEvent(new Event('input', { bubbles: true }));
        }
        // 警告メッセージを非表示（初期化時やチューニング対象が削除された時）
        if (warningDiv) {
            warningDiv.style.display = 'none';
        }
    } else {
        // チューニング対象が追加された場合、警告を非表示
        if (warningDiv) {
            warningDiv.style.display = 'none';
        }
        if (optunaState.n_trials === 1 && tunableCount > 0) {
            // チューニング対象が追加された場合、デフォルト値に戻す（ユーザーが明示的に1を設定した場合はそのまま）
            // ここでは何もしない（ユーザーの設定を尊重）
        }
    }
}

function getOptunaPayload() {
    const tunableCount = countTunableParams(paramsState);
    const config = {};
    if (optunaState.metric) config.metric = optunaState.metric;
    if (optunaState.direction) config.direction = optunaState.direction;
    
    // チューニング対象が0の場合、n_trialsを1に設定
    if (tunableCount === 0) {
        config.n_trials = 1;
    } else if (optunaState.n_trials || optunaState.n_trials === 0) {
        config.n_trials = optunaState.n_trials;
    }
    
    if (optunaState.timeout || optunaState.timeout === 0) config.timeout = optunaState.timeout;
    return config;
}

function sanitizeAugmentsState(node) {
    if (Array.isArray(node)) {
        return node
            .map(item => sanitizeAugmentsState(item))
            .filter(value => value !== undefined);
    }
    if (node && typeof node === 'object') {
        const result = {};
        Object.entries(node).forEach(([key, value]) => {
            if (value === undefined) return;
            result[key] = sanitizeAugmentsState(value);
        });
        return result;
    }
    return node;
}

function getAugmentsSchema() {
    return sanitizeAugmentsState(augumentsState || {});
}

// DOMが読み込まれた後に実行
function initializeForms() {
    if (paramsFormEl) {
        buildValueTypeMap(paramsState);
        renderParamsForm();
    }
    
    initOptunaForm();
    
    if (augumentsFormEl) {
        renderAugmentsForm();
    }
    
    // 登録済みモデルのリストボックスを初期化
    initRegisteredModelSelect();
}

// 登録済みモデルのリストボックスを初期化
function initRegisteredModelSelect() {
    const selectEl = document.getElementById('registered-model-select');
    if (!selectEl) return;
    
    // オプションを追加
    registeredModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        const label = `Run ID: ${model.mlflow_run_id.substring(0, 8)}... | Score: ${model.best_score ? model.best_score.toFixed(4) : 'N/A'} | ${model.created_at}`;
        option.textContent = label;
        selectEl.appendChild(option);
    });
    
    // 初期選択（クエリパラメータから）
    if (selectedModelId) {
        selectEl.value = selectedModelId;
    }
    
    // 変更イベント
    selectEl.addEventListener('change', async (e) => {
        const modelId = e.target.value;
        if (!modelId) {
            // 選択解除時はデフォルトのパラメータに戻す
            location.href = `/theme/${themeId}/model/development/`;
            return;
        }
        
        // 選択されたモデルのパラメータを取得
        try {
            const response = await fetch(`/api/theme/${themeId}/model/${modelId}/params/`);
            const data = await response.json();
            
            if (data.success) {
                // パラメータを適用
                Object.assign(paramsState, data.params);
                selectedModelCheckpoint = data.checkpoint_path;
                
                // フォームを再レンダリング
                buildValueTypeMap(paramsState);
                renderParamsForm();
                initOptunaForm();
                
                // Optuna設定をリセット（再学習時はn_trials=1）
                const nTrialsInput = document.getElementById('optuna-ntrials');
                if (nTrialsInput) {
                    nTrialsInput.value = '1';
                }
                
                // ページをスクロールしてフォームを表示
                document.getElementById('params-form').scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                alert(`パラメータの取得に失敗しました: ${data.error}`);
            }
        } catch (error) {
            alert(`エラーが発生しました: ${error.message}`);
        }
    });
}

// DOMContentLoadedイベントで実行
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeForms);
} else {
    // 既に読み込み完了している場合は即座に実行
    initializeForms();
}

function renderAugmentsForm() {
    if (!augumentsFormEl) return;
    augumentsFormEl.innerHTML = '';
    let hasContent = false;

    if (augumentsState.image) {
        const imageSection = createAugmentsSection('image', '画像設定');
        renderObjectFieldsForAugments(imageSection, augumentsState.image, 'image');
        augumentsFormEl.appendChild(imageSection);
        hasContent = true;
    }

    ['train', 'val', 'test'].forEach(split => {
        if (augumentsState[split]) {
            const section = createAugmentsSection(split, `${split.toUpperCase()} Augmentation`);
            renderAugmentationFields(section, augumentsState[split], split);
            augumentsFormEl.appendChild(section);
            hasContent = true;
        }
    });

    if (augumentsState.preprocessing) {
        const preprocSection = createAugmentsSection('preprocessing', '前処理設定');
        renderPreprocessingFields(preprocSection, augumentsState.preprocessing);
        augumentsFormEl.appendChild(preprocSection);
        hasContent = true;
    }

    if (augumentsState.library !== undefined) {
        const librarySection = createAugmentsSection('library', 'ライブラリ');
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';
        const label = document.createElement('label');
        label.textContent = 'ライブラリ';
        const select = document.createElement('select');
        select.className = 'form-control';
        select.innerHTML = `
            <option value="albumentations" ${augumentsState.library === 'albumentations' ? 'selected' : ''}>albumentations</option>
            <option value="torchvision" ${augumentsState.library === 'torchvision' ? 'selected' : ''}>torchvision</option>
        `;
        select.addEventListener('change', (e) => {
            augumentsState.library = e.target.value;
        });
        formGroup.appendChild(label);
        formGroup.appendChild(select);
        librarySection.appendChild(formGroup);
        augumentsFormEl.appendChild(librarySection);
        hasContent = true;
    }

    if (!hasContent) {
        const placeholder = document.createElement('p');
        placeholder.className = 'text-muted';
        placeholder.textContent = 'auguments.yaml の設定が見つかりません。';
        augumentsFormEl.appendChild(placeholder);
    }
}

function createAugmentsSection(id, title) {
    const section = document.createElement('div');
    section.className = 'augments-section';
    section.id = `augments-${id}`;
    const header = document.createElement('h3');
    header.textContent = title;
    section.appendChild(header);
    return section;
}

function renderAugmentationFields(container, augments, splitKey) {
    Object.entries(augments).forEach(([name, config]) => {
        const item = document.createElement('div');
        item.className = 'augment-item';

        if (config && typeof config === 'object' && 'enabled' in config) {
            const toggle = document.createElement('div');
            toggle.className = 'augment-toggle';
            const label = document.createElement('label');
            label.className = 'toggle-label';
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = Boolean(config.enabled);
            checkbox.addEventListener('change', (e) => {
                config.enabled = e.target.checked;
                paramsContainer.style.display = e.target.checked ? 'block' : 'none';
            });
            const span = document.createElement('span');
            span.textContent = name;
            label.appendChild(checkbox);
            label.appendChild(span);
            toggle.appendChild(label);
            item.appendChild(toggle);

            const paramsContainer = document.createElement('div');
            paramsContainer.className = 'augment-params';
            paramsContainer.style.display = config.enabled ? 'block' : 'none';
            if (config.params && Object.keys(config.params).length > 0) {
                renderObjectFieldsForAugments(paramsContainer, config.params, `${splitKey}.${name}.params`);
            } else {
                const empty = document.createElement('p');
                empty.className = 'text-muted';
                empty.textContent = '追加パラメータはありません';
                paramsContainer.appendChild(empty);
            }
            item.appendChild(paramsContainer);
        } else {
            const title = document.createElement('div');
            title.className = 'augment-label';
            title.textContent = name;
            item.appendChild(title);
            renderObjectFieldsForAugments(item, config, `${splitKey}.${name}`);
        }
        container.appendChild(item);
    });
}

function renderPreprocessingFields(container, preprocess) {
    Object.entries(preprocess).forEach(([name, config]) => {
        const item = document.createElement('div');
        item.className = 'preprocessing-item';

        if (config && typeof config === 'object' && 'enabled' in config) {
            const toggle = document.createElement('div');
            toggle.className = 'preprocessing-toggle';
            const label = document.createElement('label');
            label.className = 'toggle-label';
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = Boolean(config.enabled);
            checkbox.addEventListener('change', (e) => {
                config.enabled = e.target.checked;
                paramsContainer.style.display = e.target.checked ? 'block' : 'none';
            });
            const span = document.createElement('span');
            span.textContent = name;
            label.appendChild(checkbox);
            label.appendChild(span);
            toggle.appendChild(label);
            item.appendChild(toggle);

            const paramsContainer = document.createElement('div');
            paramsContainer.className = 'preprocessing-params';
            paramsContainer.style.display = config.enabled ? 'block' : 'none';
            if (config.params && Object.keys(config.params).length > 0) {
                renderObjectFieldsForAugments(paramsContainer, config.params, `preprocessing.${name}.params`);
            } else {
                const empty = document.createElement('p');
                empty.className = 'text-muted';
                empty.textContent = '追加パラメータはありません';
                paramsContainer.appendChild(empty);
            }
            item.appendChild(paramsContainer);
        } else {
            const title = document.createElement('div');
            title.className = 'augment-label';
            title.textContent = name;
            item.appendChild(title);
            renderObjectFieldsForAugments(item, config, `preprocessing.${name}`);
        }

        container.appendChild(item);
    });
}

function renderObjectFieldsForAugments(container, obj, path) {
    if (obj === null || obj === undefined) {
        container.appendChild(createAugmentValueField(path, null));
        return;
    }

    if (Array.isArray(obj)) {
        container.appendChild(createAugmentValueField(path, obj));
        return;
    }

    if (typeof obj !== 'object') {
        container.appendChild(createAugmentValueField(path, obj));
        return;
    }

    Object.entries(obj).forEach(([key, value]) => {
        const fieldPath = path ? `${path}.${key}` : key;
        if (value && typeof value === 'object' && !Array.isArray(value)) {
            const subgroup = document.createElement('div');
            subgroup.className = 'augment-subgroup';
            const label = document.createElement('div');
            label.className = 'augment-subgroup-title';
            label.textContent = key;
            subgroup.appendChild(label);
            renderObjectFieldsForAugments(subgroup, value, fieldPath);
            container.appendChild(subgroup);
        } else {
            container.appendChild(createAugmentValueField(fieldPath, value, key));
        }
    });
}

function createAugmentValueField(path, value, labelText) {
    const wrapper = document.createElement('div');
    wrapper.className = 'form-group';
    const label = document.createElement('label');
    label.textContent = labelText || path.split('.').slice(-1)[0];
    wrapper.appendChild(label);

    if (Array.isArray(value)) {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'form-control';
        input.value = JSON.stringify(value);
        input.placeholder = '例: [1, 2, 3]';
        input.addEventListener('change', (e) => {
            try {
                const parsed = JSON.parse(e.target.value || '[]');
                updateAugmentsState(path, parsed);
            } catch {
                alert('配列形式が正しくありません');
                e.target.value = JSON.stringify(value);
            }
        });
        wrapper.appendChild(input);
        return wrapper;
    }

    if (typeof value === 'number') {
        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'form-control';
        input.step = Number.isInteger(value) ? '1' : '0.01';
        input.value = value;
        input.addEventListener('change', (e) => {
            const parsed = Number.isInteger(value) ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
            updateAugmentsState(path, Number.isNaN(parsed) ? 0 : parsed);
        });
        wrapper.appendChild(input);
        return wrapper;
    }

    if (typeof value === 'boolean') {
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = value;
        checkbox.addEventListener('change', (e) => {
            updateAugmentsState(path, e.target.checked);
        });
        const checkboxContainer = document.createElement('div');
        checkboxContainer.className = 'toggle-label';
        checkboxContainer.appendChild(checkbox);
        checkboxContainer.appendChild(document.createTextNode('有効'));
        wrapper.appendChild(checkboxContainer);
        return wrapper;
    }

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'form-control';
    input.value = value ?? '';
    input.placeholder = value === null ? 'None' : '';
    input.addEventListener('change', (e) => {
        const val = e.target.value === '' ? null : e.target.value;
        updateAugmentsState(path, val);
    });
    wrapper.appendChild(input);
    return wrapper;
}

function updateAugmentsState(path, value) {
    if (!path) return;
    const keys = path.split('.');
    let current = augumentsState;
    for (let i = 0; i < keys.length - 1; i++) {
        const key = keys[i];
        if (current[key] === undefined || current[key] === null) {
            current[key] = {};
        }
        current = current[key];
    }
    current[keys[keys.length - 1]] = value;
}

async function saveParams() {
    const payload = {
        params_schema: sanitizeParamsState(paramsState),
        optuna_config: getOptunaPayload(),
        auguments_schema: getAugmentsSchema(),
    };

    try {
        const response = await fetch(`/api/theme/${themeId}/params/save/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!data.success) {
            alert(`保存に失敗しました: ${data.error}`);
            return false;
        }
        alert('保存しました');
        return true;
    } catch (error) {
        alert(`エラーが発生しました: ${error.message}`);
        return false;
    }
}

// 保存ボタン
document.getElementById('save-params-btn').addEventListener('click', saveParams);

// 前処理プレビューボタン
document.getElementById('preview-preprocessing-btn').addEventListener('click', async () => {
    const augmentsPayload = getAugmentsSchema();
    try {
        const response = await fetch(`/api/theme/${themeId}/preview/augmentation/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify({
                auguments_schema: augmentsPayload,
                type: 'preprocessing',
            }),
        });
        const data = await response.json();
        if (data.success && data.preview.preprocessing) {
            displayPreprocessingPreview(data.preview.preprocessing);
        } else {
            alert(`プレビュー生成に失敗しました: ${data.error || '不明なエラー'}`);
        }
    } catch (error) {
        alert(`エラーが発生しました: ${error.message}`);
    }
});

// Augmentationプレビューボタン
document.getElementById('preview-augmentation-btn').addEventListener('click', async () => {
    const augmentsPayload = getAugmentsSchema();
    try {
        const response = await fetch(`/api/theme/${themeId}/preview/augmentation/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify({
                auguments_schema: augmentsPayload,
                type: 'augmentation',
            }),
        });
        const data = await response.json();
        if (data.success && data.preview.augmentation) {
            displayAugmentationPreview(data.preview.augmentation);
        } else {
            alert(`プレビュー生成に失敗しました: ${data.error || '不明なエラー'}`);
        }
    } catch (error) {
        alert(`エラーが発生しました: ${error.message}`);
    }
});

// プレビュー描画系は既存関数を再利用
function displayPreprocessingPreview(preview) {
    const container = document.getElementById('preprocessing-images');
    container.innerHTML = '';
    const originalImages = preview.original || [];
    const preprocessedImages = preview.preprocessed || [];
    for (let i = 0; i < originalImages.length; i++) {
        const row = document.createElement('div');
        row.className = 'preview-row';
        const originalDiv = document.createElement('div');
        originalDiv.className = 'preview-item';
        originalDiv.innerHTML = `
            <h4>元画像 ${i + 1}</h4>
            <img src="${originalImages[i]}" alt="元画像 ${i + 1}">
        `;
        const processedDiv = document.createElement('div');
        processedDiv.className = 'preview-item';
        processedDiv.innerHTML = `
            <h4>前処理後 ${i + 1}</h4>
            <img src="${preprocessedImages[i]}" alt="前処理後 ${i + 1}">
        `;
        row.appendChild(originalDiv);
        row.appendChild(processedDiv);
        container.appendChild(row);
    }
    document.getElementById('preview-preprocessing').style.display = 'block';
}

function displayAugmentationPreview(preview) {
    const container = document.getElementById('augmentation-images');
    container.innerHTML = '';
    const originalImages = preview.original || [];
    const augmentedImagesList = preview.augmented || [];
    for (let i = 0; i < originalImages.length; i++) {
        const row = document.createElement('div');
        row.className = 'preview-row';
        const originalDiv = document.createElement('div');
        originalDiv.className = 'preview-item';
        originalDiv.innerHTML = `
            <h4>元画像 ${i + 1}</h4>
            <img src="${originalImages[i]}" alt="元画像 ${i + 1}">
        `;
        const augmentedDiv = document.createElement('div');
        augmentedDiv.className = 'preview-item preview-augmented';
        const augmentedImages = augmentedImagesList[i] || [];
        augmentedDiv.innerHTML = `
            <h4>Augmentation後 ${i + 1}</h4>
            <div class="augmented-samples">
                ${augmentedImages.map((img, idx) => `<img src="${img}" alt="Aug ${idx + 1}">`).join('')}
            </div>
        `;
        row.appendChild(originalDiv);
        row.appendChild(augmentedDiv);
        container.appendChild(row);
    }
    document.getElementById('preview-augmentation').style.display = 'block';
}

// 開発開始ボタン
document.getElementById('start-training-btn').addEventListener('click', async () => {
    if (!confirm('開発を開始しますか？')) {
        return;
    }
    const saved = await saveParams();
    if (!saved) return;
    try {
        const startResponse = await fetch(`/api/theme/${themeId}/training/start/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify({
                checkpoint_path: selectedModelCheckpoint || null,
            }),
        });
        const startData = await startResponse.json();
        if (startData.success) {
            window.location.href = `/theme/${themeId}/training/${startData.job_id}/`;
        } else {
            alert(`学習開始に失敗しました: ${startData.error}`);
        }
    } catch (error) {
        alert(`エラーが発生しました: ${error.message}`);
    }
});

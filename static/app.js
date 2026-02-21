(function () {
    /* ─── DOM References ─── */
    var chatArea = document.getElementById('chatArea');
    var input = document.getElementById('messageInput');
    var sendBtn = document.getElementById('sendBtn');
    var thinkingRow = document.getElementById('thinkingRow');
    var thinkingText = document.getElementById('thinkingText');
    var chipsContainer = document.getElementById('chips');
    var graphPanel = document.getElementById('graphPanel');
    var graphToggle = document.getElementById('graphToggle');
    var graphClose = document.getElementById('graphClose');
    var graphNodesEl = document.getElementById('graphNodes');

    /* ─── State ─── */
    var threadId = null;
    var isSending = false;
    var validationPassed = false;

    /* ─── Graph Node Config ─── */
    var GRAPH_NODES = [
        { id: 'router', name: 'Router', section: 'pipeline' },
        { id: 'retrieve', name: 'Retrieve', section: 'pipeline' },
        { id: 'generate', name: 'Generate', section: 'pipeline' },
        { id: 'validate', name: 'Validate', section: 'pipeline' },
        { id: 'respond', name: 'Respond', section: 'pipeline' },
        { id: 'greeting', name: 'Greeting', section: 'branch' },
        { id: 'off_topic', name: 'Off-Topic', section: 'branch' },
        { id: 'fallback', name: 'Fallback', section: 'branch' },
    ];

    var THINKING_PHASES = {
        router: 'Understanding your question',
        retrieve: 'Searching knowledge base',
        generate: 'Crafting your response',
        validate: 'Reviewing answer quality',
    };

    /* ─── Initialize Graph Panel ─── */
    function initGraphPanel() {
        graphNodesEl.innerHTML = '';
        var currentSection = '';
        for (var i = 0; i < GRAPH_NODES.length; i++) {
            var n = GRAPH_NODES[i];
            if (n.section !== currentSection) {
                currentSection = n.section;
                var label = document.createElement('div');
                label.className = 'graph-section-label';
                label.textContent = currentSection === 'pipeline' ? 'Main Pipeline' : 'Branch Nodes';
                graphNodesEl.appendChild(label);
            }
            var node = document.createElement('div');
            node.className = 'graph-node';
            node.id = 'gn-' + n.id;
            node.innerHTML =
                '<div class="node-info"><div class="node-name">' + n.name + '</div><div class="node-meta" id="gn-meta-' + n.id + '"></div></div>' +
                '<span class="node-time" id="gn-time-' + n.id + '"></span>';
            graphNodesEl.appendChild(node);
        }
    }

    function resetGraphNodes() {
        validationPassed = false;
        for (var i = 0; i < GRAPH_NODES.length; i++) {
            var el = document.getElementById('gn-' + GRAPH_NODES[i].id);
            if (el) {
                el.className = 'graph-node';
                var time = document.getElementById('gn-time-' + GRAPH_NODES[i].id);
                if (time) time.textContent = '';
                var meta = document.getElementById('gn-meta-' + GRAPH_NODES[i].id);
                if (meta) meta.textContent = '';
            }
        }
    }

    function updateGraphNode(data) {
        var el = document.getElementById('gn-' + data.node);
        if (!el) return;

        if (data.status === 'start') {
            el.className = 'graph-node active';
            graphToggle.classList.add('has-activity');
        } else if (data.status === 'complete') {
            el.className = 'graph-node complete';
            var timeEl = document.getElementById('gn-time-' + data.node);
            if (timeEl && data.duration_ms != null) {
                timeEl.textContent = data.duration_ms + 'ms';
            }
            var metaEl = document.getElementById('gn-meta-' + data.node);
            if (metaEl && data.metadata) {
                var metaText = formatNodeMeta(data.node, data.metadata);
                if (metaText) metaEl.textContent = metaText;
            }
            if (data.node === 'validate' && data.metadata && data.metadata.result === 'PASS') {
                validationPassed = true;
            }
            // Mark skipped nodes after terminal nodes
            if (data.node === 'greeting' || data.node === 'off_topic' || data.node === 'fallback') {
                markSkippedPipelineNodes();
            }
            if (data.node === 'respond') {
                markSkippedBranchNodes();
            }
        }
    }

    function formatNodeMeta(node, meta) {
        if (node === 'router' && meta.query_type) return meta.query_type + ' (' + (meta.confidence != null ? (meta.confidence * 100).toFixed(0) + '%' : '?') + ')';
        if (node === 'retrieve' && meta.doc_count != null) return meta.doc_count + ' docs retrieved';
        if (node === 'validate' && meta.result) return meta.result;
        if (node === 'respond' && meta.sources) return meta.sources.join(', ');
        return '';
    }

    function markSkippedPipelineNodes() {
        var pipeline = ['retrieve', 'generate', 'validate', 'respond'];
        for (var i = 0; i < pipeline.length; i++) {
            var el = document.getElementById('gn-' + pipeline[i]);
            if (el && el.className === 'graph-node') el.classList.add('skipped');
        }
    }

    function markSkippedBranchNodes() {
        var branches = ['greeting', 'off_topic', 'fallback'];
        for (var i = 0; i < branches.length; i++) {
            var el = document.getElementById('gn-' + branches[i]);
            if (el && el.className === 'graph-node') el.classList.add('skipped');
        }
    }

    initGraphPanel();

    /* ─── Helpers ─── */
    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function autoResizeInput() {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    }

    function formatMarkdown(text) {
        var escaped = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        escaped = escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        var lines = escaped.split('\n');
        var html = '';
        var inList = false;

        for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            if (!line) {
                if (inList) { html += '</ul>'; inList = false; }
                continue;
            }

            var bulletMatch = line.match(/^[-*]\s+(.+)/);
            var numMatch = line.match(/^\d+\.\s+(.+)/);

            if (bulletMatch || numMatch) {
                if (!inList) { html += '<ul>'; inList = true; }
                html += '<li>' + (bulletMatch ? bulletMatch[1] : numMatch[1]) + '</li>';
            } else {
                if (inList) { html += '</ul>'; inList = false; }
                html += '<p>' + line + '</p>';
            }
        }
        if (inList) html += '</ul>';
        return html || '<p>' + escaped + '</p>';
    }

    function createBotRow() {
        var row = document.createElement('div');
        row.className = 'msg-row';

        var avatar = document.createElement('div');
        avatar.className = 'bot-avatar';
        var avatarImg = document.createElement('img');
        avatarImg.src = '/assets/logo-avatar.png';
        avatarImg.alt = '7';
        avatarImg.width = 36;
        avatarImg.height = 36;
        avatar.appendChild(avatarImg);
        row.appendChild(avatar);

        var bubble = document.createElement('div');
        bubble.className = 'bubble';
        row.appendChild(bubble);

        return { row: row, bubble: bubble };
    }

    function createUserMessage(text) {
        var el = document.createElement('div');
        el.className = 'message user';
        el.textContent = text;
        return el;
    }

    function showThinking(phase) {
        thinkingText.textContent = phase || 'Thinking';
        thinkingRow.classList.add('visible');
        scrollToBottom();
    }

    function hideThinking() {
        thinkingRow.classList.remove('visible');
    }

    function updateThinkingPhase(node) {
        var phase = THINKING_PHASES[node];
        if (phase) {
            thinkingText.textContent = phase;
        }
    }

    function setDisabled(disabled) {
        isSending = disabled;
        sendBtn.disabled = disabled;
        input.disabled = disabled;
    }

    function addSourceBadges(bubbleEl, categories) {
        if (!categories || !categories.length) return;
        var div = document.createElement('div');
        div.className = 'sources';
        for (var i = 0; i < categories.length; i++) {
            var span = document.createElement('span');
            span.textContent = categories[i];
            div.appendChild(span);
        }
        bubbleEl.appendChild(div);
    }

    function addValidatedBadge(bubbleEl) {
        var badge = document.createElement('div');
        badge.className = 'validated-badge';
        badge.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" style="vertical-align:-1px;margin-right:3px"><path d="M2 6.5L4.5 9L10 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>Validated';
        bubbleEl.appendChild(badge);
    }

    /* ─── Graph Panel Toggle ─── */
    graphToggle.addEventListener('click', function () {
        graphPanel.classList.toggle('open');
        graphToggle.classList.remove('has-activity');
    });

    graphClose.addEventListener('click', function () {
        graphPanel.classList.remove('open');
    });

    /* ─── Suggestion Chips ─── */
    chipsContainer.addEventListener('click', function (e) {
        var chip = e.target.closest('.chip');
        if (!chip || isSending) return;
        var msg = chip.getAttribute('data-msg');
        if (msg) {
            input.value = msg;
            sendMessage();
            chipsContainer.style.display = 'none';
        }
    });

    /* ─── Send Message ─── */
    async function sendMessage() {
        var text = input.value.trim();
        if (!text || isSending) return;

        chatArea.insertBefore(createUserMessage(text), thinkingRow);
        input.value = '';
        input.style.height = 'auto';
        scrollToBottom();

        var bot = createBotRow();
        chatArea.insertBefore(bot.row, thinkingRow);
        var botBubble = bot.bubble;
        showThinking('Thinking');
        resetGraphNodes();
        setDisabled(true);

        var accumulated = '';
        var sources = [];
        var isStreaming = false;

        try {
            var body = { message: text };
            if (threadId) body.thread_id = threadId;

            var response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });

            if (!response.ok) {
                var errData;
                try { errData = await response.json(); } catch (_) {}
                throw new Error(
                    (errData && errData.message) || 'Request failed (' + response.status + ')'
                );
            }

            var reader = response.body.getReader();
            var decoder = new TextDecoder();
            var buffer = '';

            while (true) {
                var result = await reader.read();
                if (result.done) break;

                buffer += decoder.decode(result.value, { stream: true });
                var lines = buffer.split('\n');
                buffer = lines.pop();

                var currentEvent = '';
                for (var i = 0; i < lines.length; i++) {
                    var line = lines[i];

                    if (line.startsWith('event: ')) {
                        currentEvent = line.substring(7).trim();
                        continue;
                    }

                    if (line.startsWith('data: ')) {
                        var dataStr = line.substring(6);
                        var data;
                        try { data = JSON.parse(dataStr); } catch (_) { continue; }

                        if (currentEvent === 'metadata' && data.thread_id) {
                            threadId = data.thread_id;
                        }
                        else if (currentEvent === 'graph_node') {
                            updateGraphNode(data);
                            if (data.status === 'start') {
                                updateThinkingPhase(data.node);
                            }
                        }
                        else if (currentEvent === 'token' && data.content) {
                            if (!isStreaming) {
                                hideThinking();
                                isStreaming = true;
                                botBubble.classList.add('streaming-cursor');
                            }
                            accumulated += data.content;
                            botBubble.innerHTML = formatMarkdown(accumulated);
                            scrollToBottom();
                        }
                        else if (currentEvent === 'replace' && data.content) {
                            hideThinking();
                            accumulated = data.content;
                            botBubble.innerHTML = formatMarkdown(accumulated);
                            scrollToBottom();
                        }
                        else if (currentEvent === 'sources' && data.sources) {
                            sources = data.sources;
                        }
                        else if (currentEvent === 'error' || data.error) {
                            hideThinking();
                            accumulated = data.detail || data.error || data.message || 'Something went wrong.';
                            botBubble.innerHTML = formatMarkdown(accumulated);
                        }
                        else if (currentEvent === 'done') {
                            // Stream complete
                        }
                        else if (data.response) {
                            hideThinking();
                            accumulated = data.response;
                            botBubble.innerHTML = formatMarkdown(accumulated);
                            if (data.thread_id) threadId = data.thread_id;
                            if (data.sources) sources = data.sources;
                            scrollToBottom();
                        }

                        currentEvent = '';
                    }
                }
            }

            botBubble.classList.remove('streaming-cursor');
            addSourceBadges(botBubble, sources);
            if (validationPassed) {
                addValidatedBadge(botBubble);
            }
            graphToggle.classList.remove('has-activity');

            if (!accumulated) {
                botBubble.innerHTML = formatMarkdown('I received your message but had trouble generating a response. Please try again.');
            }

        } catch (err) {
            hideThinking();
            botBubble.classList.remove('streaming-cursor');
            botBubble.innerHTML = formatMarkdown(
                'Sorry, I encountered an error: ' + err.message + '. Please try again.'
            );
        } finally {
            hideThinking();
            setDisabled(false);
            input.focus();
            scrollToBottom();
        }
    }

    /* ─── Event Listeners ─── */
    sendBtn.addEventListener('click', sendMessage);

    input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    input.addEventListener('input', autoResizeInput);

    scrollToBottom();
})();

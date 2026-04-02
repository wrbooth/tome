(function () {
    'use strict';

    // ── DOM refs ─────────────────────────────────────────────────────────
    var form = document.getElementById('search-form');
    var input = document.getElementById('query-input');
    var submitBtn = document.getElementById('submit-btn');
    var chatMessages = document.getElementById('chat-messages');
    var chatScroll = chatMessages; // the scrollable container
    var welcome = document.getElementById('welcome');
    var newChatBtn = document.getElementById('new-chat-btn');

    // ── State ────────────────────────────────────────────────────────────
    var isStreaming = false;

    // ── Utilities ────────────────────────────────────────────────────────

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function scrollToBottom() {
        chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    // ── SSE Parser ───────────────────────────────────────────────────────
    // Cannot use EventSource for POST requests; manually parse SSE frames.

    function createSSEParser(onEvent) {
        var buffer = '';
        var currentEvent = '';
        var currentData = '';

        return {
            feed: function (chunk) {
                buffer += chunk;
                var lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete last line

                for (var i = 0; i < lines.length; i++) {
                    var line = lines[i];
                    if (line.startsWith('event: ')) {
                        currentEvent = line.slice(7);
                    } else if (line.startsWith('data: ')) {
                        currentData = line.slice(6);
                    } else if (line === '') {
                        if (currentEvent) {
                            onEvent(currentEvent, currentData);
                        }
                        currentEvent = '';
                        currentData = '';
                    }
                }
            },
            flush: function () {
                if (currentEvent && currentData) {
                    onEvent(currentEvent, currentData);
                }
            }
        };
    }

    // ── DOM helpers ──────────────────────────────────────────────────────

    function getMessageContainer() {
        return chatMessages.querySelector('.max-w-3xl');
    }

    function appendUserMessage(text) {
        var container = getMessageContainer();
        var div = document.createElement('div');
        div.className = 'flex justify-end';
        div.innerHTML =
            '<div class="bg-blue-600 text-white rounded-2xl rounded-br-sm px-4 py-2.5 max-w-lg text-sm">' +
            escapeHtml(text) +
            '</div>';
        container.appendChild(div);
        scrollToBottom();
    }

    function createAssistantMessage() {
        var container = getMessageContainer();
        var wrapper = document.createElement('div');
        wrapper.className = 'space-y-3';

        var bubble = document.createElement('div');
        bubble.className = 'bg-white border border-slate-200 rounded-2xl rounded-bl-sm px-5 py-4 max-w-2xl shadow-sm';

        var typingEl = document.createElement('div');
        typingEl.className = 'flex gap-1.5 py-1';
        typingEl.innerHTML =
            '<span class="typing-dot"></span>' +
            '<span class="typing-dot"></span>' +
            '<span class="typing-dot"></span>';

        var contentEl = document.createElement('div');
        contentEl.className = 'prose text-sm hidden';

        bubble.appendChild(typingEl);
        bubble.appendChild(contentEl);
        wrapper.appendChild(bubble);
        container.appendChild(wrapper);
        scrollToBottom();

        return { wrapper: wrapper, contentEl: contentEl, typingEl: typingEl };
    }

    // ── Result cards (Task 5) ────────────────────────────────────────────

    function renderResultCards(wrapper, searchResults) {
        if (!searchResults || !searchResults.results || !searchResults.results.length) return;

        var sorted = searchResults.results.slice().sort(function (a, b) {
            return b.score - a.score;
        });

        var cardsEl = document.createElement('div');
        cardsEl.className = 'space-y-1.5 mt-3';

        var heading = document.createElement('p');
        heading.className = 'text-xs font-medium text-slate-400 uppercase tracking-wide mb-2';
        heading.textContent = sorted.length + ' source' + (sorted.length !== 1 ? 's' : '') + ' found';
        cardsEl.appendChild(heading);

        for (var i = 0; i < sorted.length; i++) {
            var r = sorted[i];
            var card = document.createElement('details');
            card.className = 'result-card border border-slate-200 rounded-lg bg-white overflow-hidden';

            var breadcrumb = '';
            if (r.headings_path && r.headings_path.length) {
                breadcrumb = r.headings_path.map(function (h) { return escapeHtml(h); }).join(' &#8250; ');
            }

            var scorePercent = Math.round(r.score * 100);

            card.innerHTML =
                '<summary class="flex items-start gap-3 px-4 py-2.5 cursor-pointer hover:bg-slate-50 select-none text-sm">' +
                    '<span class="chevron text-slate-400 mt-0.5 text-xs shrink-0">&#9654;</span>' +
                    '<div class="flex-1 min-w-0">' +
                        '<div class="font-medium text-slate-700 truncate">' + escapeHtml(r.title) + '</div>' +
                        '<div class="flex items-center gap-2 mt-0.5 text-xs text-slate-400">' +
                            '<span>p.\u00a0' + r.page + '</span>' +
                            (breadcrumb ? '<span class="text-slate-300">|</span><span class="truncate">' + breadcrumb + '</span>' : '') +
                        '</div>' +
                    '</div>' +
                    '<div class="flex items-center gap-2 shrink-0 mt-1">' +
                        '<div class="w-12 h-1 bg-slate-200 rounded-full overflow-hidden">' +
                            '<div class="h-full bg-blue-500 rounded-full" style="width:' + scorePercent + '%"></div>' +
                        '</div>' +
                        '<span class="text-xs text-slate-400 tabular-nums">.' + String(scorePercent).padStart(2, '0') + '</span>' +
                    '</div>' +
                '</summary>' +
                '<div class="px-4 pb-3 text-xs text-slate-600 border-t border-slate-100 pt-2 ml-6">' +
                    '<p class="whitespace-pre-wrap leading-relaxed">' + escapeHtml(r.text) + '</p>' +
                '</div>';

            cardsEl.appendChild(card);
        }

        wrapper.appendChild(cardsEl);
        scrollToBottom();
    }

    // ── Source citations (Task 4) ────────────────────────────────────────

    function processSourceCitations(contentEl, searchResults) {
        if (!searchResults || !searchResults.results || !searchResults.results.length) return;

        // Build lookup: "title, Page N" (lowercased) -> passage
        var lookup = {};
        for (var i = 0; i < searchResults.results.length; i++) {
            var r = searchResults.results[i];
            var key = (r.title + ', Page ' + r.page).toLowerCase();
            lookup[key] = r;
            // Also match "title, p. N" and "title, page N"
            var key2 = (r.title + ', p. ' + r.page).toLowerCase();
            lookup[key2] = r;
        }

        var html = contentEl.innerHTML;
        // Match [Document Title, Page X] or [Document Title, p. X] patterns
        var citationRegex = /\[([^\]]+?,\s*(?:Page|p\.)\s+\d+)\]/gi;

        contentEl.innerHTML = html.replace(citationRegex, function (match, inner) {
            var passage = lookup[inner.toLowerCase()];
            if (!passage) return match;

            return '<details class="source-citation inline">' +
                '<summary>[' + escapeHtml(inner) + ']</summary>' +
                '<blockquote class="mt-2 pl-3 border-l-2 border-blue-300 text-xs text-slate-600 bg-slate-50 rounded p-2 block">' +
                escapeHtml(passage.text) +
                '</blockquote>' +
                '</details>';
        });
    }

    // ── Search handler ───────────────────────────────────────────────────

    form.addEventListener('submit', handleSearch);

    async function handleSearch(e) {
        e.preventDefault();
        var query = input.value.trim();
        if (!query || isStreaming) return;

        isStreaming = true;
        submitBtn.disabled = true;

        // Remove welcome message
        if (welcome) {
            welcome.remove();
            welcome = null;
        }

        appendUserMessage(query);
        input.value = '';

        var msg = createAssistantMessage();
        var rawText = '';
        var tokenBuffer = '';
        var rafPending = false;
        var searchResults = null;
        var firstToken = true;

        function flushTokens() {
            if (!tokenBuffer) { rafPending = false; return; }

            // Hide typing indicator, show content
            if (firstToken) {
                msg.typingEl.classList.add('hidden');
                msg.contentEl.classList.remove('hidden');
                msg.contentEl.classList.add('streaming-cursor');
                firstToken = false;
            }

            rawText += tokenBuffer;
            tokenBuffer = '';
            rafPending = false;

            msg.contentEl.innerHTML = DOMPurify.sanitize(marked.parse(rawText));
            scrollToBottom();
        }

        var parser = createSSEParser(function (event, data) {
            switch (event) {
                case 'search_results':
                    try {
                        searchResults = JSON.parse(data);
                        renderResultCards(msg.wrapper, searchResults);
                    } catch (e) { /* ignore parse error */ }
                    break;

                case 'token':
                    // _sse_event sends strings as bare data (not JSON-encoded)
                    tokenBuffer += data;
                    if (!rafPending) {
                        rafPending = true;
                        requestAnimationFrame(flushTokens);
                    }
                    break;

                case 'done':
                    flushTokens();
                    msg.contentEl.classList.remove('streaming-cursor');
                    if (!firstToken) {
                        processSourceCitations(msg.contentEl, searchResults);
                    }
                    isStreaming = false;
                    submitBtn.disabled = false;
                    input.focus();
                    break;

                case 'error':
                    msg.typingEl.classList.add('hidden');
                    msg.contentEl.classList.remove('hidden');
                    try {
                        var err = JSON.parse(data);
                        msg.contentEl.textContent = 'Error: ' + err.message;
                    } catch (e) {
                        msg.contentEl.textContent = 'Error: ' + data;
                    }
                    msg.contentEl.classList.add('text-red-600');
                    isStreaming = false;
                    submitBtn.disabled = false;
                    break;
            }
        });

        try {
            var resp = await fetch('/api/search/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query }),
            });

            if (!resp.ok) {
                throw new Error('Server returned ' + resp.status);
            }

            var reader = resp.body.getReader();
            var decoder = new TextDecoder();

            while (true) {
                var result = await reader.read();
                if (result.done) break;
                parser.feed(decoder.decode(result.value, { stream: true }));
            }
            parser.flush();
        } catch (err) {
            msg.typingEl.classList.add('hidden');
            msg.contentEl.classList.remove('hidden');
            msg.contentEl.textContent = 'Connection error: ' + err.message;
            msg.contentEl.classList.add('text-red-600');
            isStreaming = false;
            submitBtn.disabled = false;
        }
    }

    // ── New chat ─────────────────────────────────────────────────────────

    newChatBtn.addEventListener('click', function () {
        var container = getMessageContainer();
        container.innerHTML =
            '<div id="welcome" class="text-center mt-24 space-y-3">' +
                '<div class="text-4xl">&#128218;</div>' +
                '<p class="text-lg text-slate-500">Ask a question about the indexed documents</p>' +
                '<p class="text-sm text-slate-400">Results are drawn from historical records, deeds, and local history materials.</p>' +
            '</div>';
        welcome = document.getElementById('welcome');
        input.focus();
    });

    // ── Keyboard shortcuts ───────────────────────────────────────────────

    input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            // Default form submit handles this
        }
    });

    // Focus input on load
    input.focus();

})();

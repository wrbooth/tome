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
    var sourcePanel = document.getElementById('source-panel');
    var sourcePanelClose = document.getElementById('source-panel-close');
    var sourceList = document.getElementById('source-list');

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
        return chatMessages.querySelector('.max-w-4xl');
    }

    function appendUserMessage(text) {
        var container = getMessageContainer();
        var div = document.createElement('div');
        div.className = 'flex justify-end';
        div.innerHTML =
            '<div class="bg-blue-600 text-white rounded-2xl rounded-br-sm px-5 py-3 max-w-xl text-base">' +
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
        bubble.className = 'max-w-none';

        var typingEl = document.createElement('div');
        typingEl.className = 'flex gap-1.5 py-1';
        typingEl.innerHTML =
            '<span class="typing-dot"></span>' +
            '<span class="typing-dot"></span>' +
            '<span class="typing-dot"></span>';

        var contentEl = document.createElement('div');
        contentEl.className = 'prose text-base hidden';

        bubble.appendChild(typingEl);
        bubble.appendChild(contentEl);
        wrapper.appendChild(bubble);
        container.appendChild(wrapper);
        scrollToBottom();

        return { wrapper: wrapper, contentEl: contentEl, typingEl: typingEl };
    }

    // ── Source panel ─────────────────────────────────────────────────────

    // Stores the citation map for the current answer: { "title, page N" -> citationIndex }
    var citationMap = {};
    var citationList = []; // ordered array of { title, page, passage } for panel rendering

    function renderSourcePanel(searchResults) {
        if (!searchResults || !searchResults.results || !searchResults.results.length) return;

        // Build source cards in the panel (not in chat)
        // We'll assign citation numbers later in processSourceCitations,
        // but pre-render all results so they're ready
        sourceList.innerHTML = '';
        var results = searchResults.results;

        // Build lookup by "title, page" for citation matching
        var lookup = {};
        for (var i = 0; i < results.length; i++) {
            var r = results[i];
            var key = r.title.toLowerCase() + ', page ' + r.page;
            if (!lookup[key]) {
                lookup[key] = r;
            }
            // Also store with "p." variant
            var key2 = r.title.toLowerCase() + ', p. ' + r.page;
            if (!lookup[key2]) {
                lookup[key2] = r;
            }
        }

        // Store lookup for citation processing
        window.Tome = window.Tome || {};
        window.Tome._sourceLookup = lookup;
        window.Tome._sourceResults = results;
    }

    function buildSourcePanelCards() {
        // Called after processSourceCitations has built citationList
        sourceList.innerHTML = '';

        if (!citationList.length) return;

        for (var i = 0; i < citationList.length; i++) {
            var src = citationList[i];
            var card = document.createElement('div');
            card.className = 'source-card px-4 py-4';
            card.setAttribute('data-citation-index', String(i + 1));

            var breadcrumb = '';
            if (src.headings_path && src.headings_path.length) {
                breadcrumb = src.headings_path.map(function (h) { return escapeHtml(h); }).join(' > ');
            }

            card.innerHTML =
                '<div class="flex items-start gap-2 mb-2">' +
                    '<span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-700 text-xs font-bold shrink-0">' + (i + 1) + '</span>' +
                    '<div class="min-w-0">' +
                        '<div class="text-sm font-semibold text-slate-700">' + escapeHtml(src.title) + '</div>' +
                        '<div class="text-sm text-slate-400">Page ' + src.page +
                            (breadcrumb ? ' &middot; ' + breadcrumb : '') +
                        '</div>' +
                    '</div>' +
                '</div>' +
                '<p class="text-sm text-slate-600 leading-relaxed whitespace-pre-wrap">' + escapeHtml(src.text) + '</p>';

            sourceList.appendChild(card);
        }
    }

    function openSourcePanel(citationIndex) {
        sourcePanel.classList.remove('source-panel-closed');
        sourcePanel.classList.add('source-panel-open');

        // Highlight the active card
        var cards = sourceList.querySelectorAll('.source-card');
        for (var i = 0; i < cards.length; i++) {
            cards[i].classList.remove('source-active');
        }
        var target = sourceList.querySelector('[data-citation-index="' + citationIndex + '"]');
        if (target) {
            target.classList.add('source-active');
            target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    function closeSourcePanel() {
        sourcePanel.classList.remove('source-panel-open');
        sourcePanel.classList.add('source-panel-closed');
    }

    sourcePanelClose.addEventListener('click', closeSourcePanel);

    // ── Source citations → superscript numbers ───────────────────────────
    // Process citations in RAW TEXT (before markdown rendering) so that
    // marked.parse() doesn't interfere with the bracket syntax.

    function processSourceCitations(rawText, searchResults) {
        if (!searchResults || !searchResults.results || !searchResults.results.length) {
            return rawText;
        }

        var lookup = (window.Tome && window.Tome._sourceLookup) || {};

        // Reset citation tracking
        citationMap = {};
        citationList = [];
        var nextNum = 1;

        // Match [Document Title, Page X] or [Document Title, p. X] patterns
        // Also handle escaped brackets \[...\] that LLMs sometimes produce in markdown
        // Also handle multi-page: [Title, Page X; Page Y]
        var citationRegex = /\\?\[([^\]]+?,\s*(?:Page|p\.)\s+\d+(?:\s*;\s*[^\]]*?\d+)*)\\?\]/gi;

        var processed = rawText.replace(citationRegex, function (match, inner) {
            // Handle potential multi-citation by taking the first page reference
            var pageMatch = inner.match(/,\s*(?:Page|p\.)\s+(\d+)/i);
            if (!pageMatch) return match;

            var key = inner.toLowerCase().replace(/\s+/g, ' ').trim();
            // Also try a simplified key (just title + first page)
            var titlePart = inner.split(',')[0].trim();
            var simpleKey = titlePart.toLowerCase() + ', page ' + pageMatch[1];

            // Determine citation number (reuse for same key)
            var num;
            var lookupKey = citationMap[key] != null ? key : (citationMap[simpleKey] != null ? simpleKey : null);

            if (lookupKey != null) {
                num = citationMap[lookupKey];
            } else {
                num = nextNum++;
                citationMap[key] = num;
                citationMap[simpleKey] = num;

                var passage = lookup[simpleKey] || lookup[key];
                citationList.push({
                    title: passage ? passage.title : titlePart,
                    page: passage ? passage.page : parseInt(pageMatch[1]),
                    text: passage ? passage.text : '',
                    headings_path: passage ? (passage.headings_path || []) : [],
                    score: passage ? passage.score : 0,
                });
            }

            // Return raw HTML that marked will pass through as inline HTML
            return '<sup class="source-ref" data-citation="' + num + '">' + num + '</sup>';
        });

        return processed;
    }

    function bindCitationClicks(contentEl) {
        var refs = contentEl.querySelectorAll('.source-ref');
        for (var j = 0; j < refs.length; j++) {
            refs[j].addEventListener('click', function () {
                var idx = this.getAttribute('data-citation');
                openSourcePanel(idx);
            });
        }
    }

    // ── Search handler ───────────────────────────────────────────────────

    form.addEventListener('submit', handleSearch);

    async function handleSearch(e) {
        e.preventDefault();
        var query = input.value.trim();
        if (!query || isStreaming) return;

        isStreaming = true;
        submitBtn.disabled = true;

        // Close source panel from previous search
        closeSourcePanel();

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
                        renderSourcePanel(searchResults);
                    } catch (e) { /* ignore parse error */ }
                    break;

                case 'token':
                    try { tokenBuffer += JSON.parse(data); }
                    catch (e) { tokenBuffer += data; }
                    if (!rafPending) {
                        rafPending = true;
                        requestAnimationFrame(flushTokens);
                    }
                    break;

                case 'done':
                    // Flush remaining tokens into rawText
                    if (tokenBuffer) {
                        rawText += tokenBuffer;
                        tokenBuffer = '';
                    }
                    rafPending = false;

                    // Process citations in raw text BEFORE markdown rendering
                    var finalText = rawText;
                    if (!firstToken && searchResults) {
                        finalText = processSourceCitations(rawText, searchResults);
                    }

                    // Final render with citations converted to <sup> tags
                    msg.typingEl.classList.add('hidden');
                    msg.contentEl.classList.remove('hidden', 'streaming-cursor');
                    msg.contentEl.innerHTML = DOMPurify.sanitize(
                        marked.parse(finalText),
                        { ADD_ATTR: ['data-citation'] }
                    );

                    // Build source panel and bind click handlers
                    buildSourcePanelCards();
                    bindCitationClicks(msg.contentEl);

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
                body: JSON.stringify({
                    query: query,
                    document_id: window.Tome && window.Tome.activeDocumentId || null,
                }),
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
        // Clear document filter if active
        if (window.Tome && window.Tome.clearDocumentFilter) {
            window.Tome.clearDocumentFilter();
        }
        // Close source panel
        closeSourcePanel();

        var container = getMessageContainer();
        container.innerHTML =
            '<div id="welcome" class="text-center mt-20 space-y-4">' +
                '<div class="text-5xl">&#128218;</div>' +
                '<p class="text-xl text-slate-600 font-medium">Ask a question about the indexed documents</p>' +
                '<p class="text-base text-slate-400">Results are drawn from historical records, deeds, and local history materials.</p>' +
                '<div class="pt-4 flex flex-wrap justify-center gap-2">' +
                    '<button class="suggested-query text-sm px-4 py-2 rounded-full border border-slate-200 text-slate-500 hover:text-slate-700 hover:border-slate-300 hover:bg-white transition-colors">Who were the earliest settlers?</button>' +
                    '<button class="suggested-query text-sm px-4 py-2 rounded-full border border-slate-200 text-slate-500 hover:text-slate-700 hover:border-slate-300 hover:bg-white transition-colors">What industries were prominent?</button>' +
                    '<button class="suggested-query text-sm px-4 py-2 rounded-full border border-slate-200 text-slate-500 hover:text-slate-700 hover:border-slate-300 hover:bg-white transition-colors">Tell me about the founding of Cambridge</button>' +
                '</div>' +
            '</div>';
        welcome = document.getElementById('welcome');
        bindSuggestedQueries();
        input.focus();
    });

    // ── Keyboard shortcuts ───────────────────────────────────────────────

    input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            // Default form submit handles this
        }
    });

    // ── Suggested queries ───────────────────────────────────────────────

    function bindSuggestedQueries() {
        var btns = document.querySelectorAll('.suggested-query');
        for (var i = 0; i < btns.length; i++) {
            btns[i].addEventListener('click', function () {
                input.value = this.textContent;
                form.dispatchEvent(new Event('submit'));
            });
        }
    }

    bindSuggestedQueries();

    // Focus input on load
    input.focus();

})();

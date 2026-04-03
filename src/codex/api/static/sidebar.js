(function () {
    'use strict';

    // ── Shared state ─────────────────────────────────────────────────────
    window.Codex = window.Codex || {};
    window.Codex.activeDocumentId = null;
    window.Codex.activeDocumentTitle = null;

    // ── DOM refs ─────────────────────────────────────────────────────────
    var sidebar = document.getElementById('sidebar');
    var sidebarToggle = document.getElementById('sidebar-toggle');
    var docList = document.getElementById('doc-list');
    var docCount = document.getElementById('doc-count');
    var docCountLabel = document.getElementById('doc-count-label');
    var docSearch = document.getElementById('doc-search');
    var activeFilters = document.getElementById('active-filters');
    var uploadBtn = document.getElementById('upload-btn');
    var uploadModal = document.getElementById('upload-modal');
    var uploadModalClose = document.getElementById('upload-modal-close');
    var uploadFileInput = document.getElementById('upload-file');
    var uploadFileName = document.getElementById('upload-file-name');
    var uploadSubmitBtn = document.getElementById('upload-submit');
    var uploadCancelBtn = document.getElementById('upload-cancel');
    var uploadStatus = document.getElementById('upload-status');
    var dropZone = document.getElementById('drop-zone');

    // ── State ────────────────────────────────────────────────────────────
    var documents = [];
    var filterTimer = null;
    var currentView = 'list'; // 'list' or 'detail'

    // ── Sidebar toggle ───────────────────────────────────────────────────

    function isSidebarOpen() {
        return !sidebar.classList.contains('collapsed');
    }

    function setSidebarOpen(open) {
        if (open) {
            sidebar.classList.remove('collapsed');
            sidebarToggle.title = 'Hide sidebar';
        } else {
            sidebar.classList.add('collapsed');
            sidebarToggle.title = 'Show sidebar';
        }
        localStorage.setItem('codex-sidebar-open', open ? '1' : '0');
    }

    sidebarToggle.addEventListener('click', function () {
        setSidebarOpen(!isSidebarOpen());
    });

    // Restore sidebar state from localStorage (default: open)
    var savedState = localStorage.getItem('codex-sidebar-open');
    if (savedState === '0') {
        setSidebarOpen(false);
    }

    // ── Fetch documents ──────────────────────────────────────────────────

    async function fetchDocuments() {
        try {
            var resp = await fetch('/api/documents');
            if (!resp.ok) throw new Error('Failed to fetch documents');
            documents = await resp.json();
            renderDocumentList();
        } catch (err) {
            docList.innerHTML =
                '<p class="text-sm text-red-500 p-4">Failed to load documents.</p>';
            updateDocCount(0);
        }
    }

    // ── Helper: update document count with correct singular/plural ───────

    function updateDocCount(count) {
        docCount.textContent = String(count);
        docCountLabel.textContent = count === 1 ? 'document' : 'documents';
    }

    // ── Render document list ─────────────────────────────────────────────

    function renderDocumentList(filterText) {
        currentView = 'list';
        var filtered = documents;
        if (filterText) {
            var q = filterText.toLowerCase();
            filtered = documents.filter(function (d) {
                return d.title.toLowerCase().includes(q) ||
                       (d.authors && d.authors.join(' ').toLowerCase().includes(q));
            });
        }

        updateDocCount(filtered.length);

        if (!filtered.length) {
            docList.innerHTML = documents.length
                ? '<p class="text-sm text-slate-400 p-4 text-center">No matching documents.</p>'
                : '<p class="text-sm text-slate-400 p-4 text-center">No documents indexed yet.</p>';
            return;
        }

        var html = '';
        for (var i = 0; i < filtered.length; i++) {
            var d = filtered[i];
            var isActive = window.Codex.activeDocumentId === d.id;
            var activeClass = isActive
                ? 'border-l-3 border-l-blue-500 bg-blue-50'
                : 'border-l-3 border-l-transparent hover:bg-slate-50';

            var authors = d.authors && d.authors.length
                ? escapeHtml(d.authors.join(', '))
                : '';
            var year = d.pub_year ? String(d.pub_year) : '';
            var meta = [authors, year].filter(Boolean).join(' &middot; ');

            html +=
                '<div class="doc-item px-4 py-3 cursor-pointer ' + activeClass + '" data-doc-id="' + d.id + '">' +
                    '<div class="flex items-start justify-between gap-2">' +
                        '<div class="min-w-0 flex-1">' +
                            '<div class="text-base font-medium text-slate-700 leading-snug line-clamp-2">' + escapeHtml(d.title) + '</div>' +
                            (meta ? '<div class="text-sm text-slate-400 mt-1 truncate">' + meta + '</div>' : '') +
                        '</div>' +
                        '<button class="doc-info-btn shrink-0 mt-0.5 p-1.5 rounded-md hover:bg-slate-200 text-slate-300 hover:text-slate-600 transition-colors opacity-0 group-item-hover:opacity-100" data-doc-id="' + d.id + '" title="View details">' +
                            '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                                '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>' +
                            '</svg>' +
                        '</button>' +
                    '</div>' +
                '</div>';
        }

        docList.innerHTML = html;

        // Attach click handlers
        var items = docList.querySelectorAll('.doc-item');
        for (var j = 0; j < items.length; j++) {
            items[j].addEventListener('click', handleDocItemClick);
        }
        var infoBtns = docList.querySelectorAll('.doc-info-btn');
        for (var k = 0; k < infoBtns.length; k++) {
            infoBtns[k].addEventListener('click', handleDocInfoClick);
        }
    }

    // ── Document item click → set filter ─────────────────────────────────

    function handleDocItemClick(e) {
        // Don't filter if they clicked the info button
        if (e.target.closest('.doc-info-btn')) return;

        var docId = this.getAttribute('data-doc-id');
        var doc = documents.find(function (d) { return d.id === docId; });
        if (!doc) return;

        if (window.Codex.activeDocumentId === docId) {
            clearDocumentFilter();
        } else {
            setDocumentFilter(docId, doc.title);
        }
    }

    function setDocumentFilter(docId, title) {
        window.Codex.activeDocumentId = docId;
        window.Codex.activeDocumentTitle = title;
        renderFilterChip();
        renderDocumentList(docSearch.value.trim());
    }

    function clearDocumentFilter() {
        window.Codex.activeDocumentId = null;
        window.Codex.activeDocumentTitle = null;
        renderFilterChip();
        renderDocumentList(docSearch.value.trim());
    }

    // Expose for app.js (New Chat)
    window.Codex.clearDocumentFilter = clearDocumentFilter;

    // ── Filter chip ──────────────────────────────────────────────────────

    function renderFilterChip() {
        if (!window.Codex.activeDocumentId) {
            activeFilters.classList.add('hidden');
            activeFilters.innerHTML = '';
            return;
        }

        activeFilters.classList.remove('hidden');
        activeFilters.innerHTML =
            '<span class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full ' +
                'bg-blue-50 text-blue-700 text-sm font-medium border border-blue-200 max-w-full">' +
                '<svg class="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                    '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" ' +
                        'd="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>' +
                '</svg>' +
                '<span class="truncate">' + escapeHtml(window.Codex.activeDocumentTitle) + '</span>' +
                '<button id="clear-filter-btn" class="ml-0.5 hover:text-blue-900 shrink-0 text-lg leading-none" title="Remove filter">&times;</button>' +
            '</span>';

        document.getElementById('clear-filter-btn').addEventListener('click', function (e) {
            e.stopPropagation();
            clearDocumentFilter();
        });
    }

    // ── Document detail view ─────────────────────────────────────────────

    function handleDocInfoClick(e) {
        e.stopPropagation();
        var docId = this.getAttribute('data-doc-id');
        showDocumentDetail(docId);
    }

    async function showDocumentDetail(docId) {
        currentView = 'detail';
        docList.innerHTML =
            '<div class="p-6 text-center">' +
                '<div class="flex gap-1.5 justify-center py-4">' +
                    '<span class="typing-dot"></span>' +
                    '<span class="typing-dot"></span>' +
                    '<span class="typing-dot"></span>' +
                '</div>' +
            '</div>';

        try {
            var resp = await fetch('/api/documents/' + docId);
            if (!resp.ok) throw new Error('Failed to fetch document details');
            var detail = await resp.json();
            renderDocumentDetail(detail);
        } catch (err) {
            docList.innerHTML =
                '<div class="p-4">' +
                    '<button class="back-to-list text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1 mb-3">' +
                        '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>' +
                        '</svg>' +
                        'All Documents' +
                    '</button>' +
                    '<p class="text-sm text-red-500">Failed to load document details.</p>' +
                '</div>';
            docList.querySelector('.back-to-list').addEventListener('click', function () {
                renderDocumentList(docSearch.value.trim());
            });
        }
    }

    function renderDocumentDetail(detail) {
        var doc = detail.document;
        var authors = doc.authors && doc.authors.length ? doc.authors.join(', ') : 'Unknown';
        var year = doc.pub_year || 'N/A';
        var pages = detail.pages;
        var pageRange = (pages.min_page != null && pages.max_page != null)
            ? pages.min_page + '–' + pages.max_page
            : 'N/A';
        var isActive = window.Codex.activeDocumentId === doc.id;

        docList.innerHTML =
            '<div class="p-4 space-y-4">' +
                // Back button
                '<button class="back-to-list text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1">' +
                    '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>' +
                    '</svg>' +
                    'All Documents' +
                '</button>' +

                // Title & meta
                '<div>' +
                    '<h3 class="text-base font-semibold text-slate-800 leading-snug">' + escapeHtml(doc.title) + '</h3>' +
                    '<p class="text-sm text-slate-400 mt-1">' + escapeHtml(authors) + ' &middot; ' + escapeHtml(String(year)) + '</p>' +
                '</div>' +

                // Stats grid
                '<div class="grid grid-cols-2 gap-2.5">' +
                    statCard('Passages', detail.passages.total_passages) +
                    statCard('Embedded', detail.passages.embedded_passages) +
                    statCard('Pages', pageRange) +
                    statCard('Entities', detail.entities.entity_count) +
                    statCard('Years referenced', detail.years.year_count) +
                '</div>' +

                // Filter button
                '<button class="detail-filter-btn w-full text-sm font-medium py-2.5 px-3 rounded-md transition-colors ' +
                    (isActive
                        ? 'bg-blue-100 text-blue-700 border border-blue-300 hover:bg-blue-200'
                        : 'bg-blue-600 text-white hover:bg-blue-700') +
                '" data-doc-id="' + doc.id + '">' +
                    (isActive ? 'Currently filtering' : 'Filter searches to this document') +
                '</button>' +

                // Delete button
                '<button class="detail-delete-btn w-full text-sm font-medium py-2.5 px-3 rounded-md ' +
                    'text-red-600 border border-red-200 hover:bg-red-50 transition-colors" ' +
                    'data-doc-id="' + doc.id + '">' +
                    'Delete document' +
                '</button>' +
            '</div>';

        // Attach handlers
        docList.querySelector('.back-to-list').addEventListener('click', function () {
            renderDocumentList(docSearch.value.trim());
        });
        docList.querySelector('.detail-filter-btn').addEventListener('click', function () {
            var id = this.getAttribute('data-doc-id');
            if (window.Codex.activeDocumentId === id) {
                clearDocumentFilter();
            } else {
                setDocumentFilter(id, doc.title);
            }
            // Re-render detail to update button state
            showDocumentDetail(id);
        });

        docList.querySelector('.detail-delete-btn').addEventListener('click', function () {
            var id = this.getAttribute('data-doc-id');
            deleteDocument(id, doc.title);
        });
    }

    async function deleteDocument(docId, title) {
        if (!confirm('Delete "' + title + '"? This will remove all passages, entities, and years. This cannot be undone.')) {
            return;
        }

        try {
            var resp = await fetch('/api/documents/' + docId, { method: 'DELETE' });
            if (!resp.ok) {
                var err = await resp.json().catch(function () { return {}; });
                throw new Error(err.detail || 'Delete failed');
            }

            // Clear filter if we just deleted the active document
            if (window.Codex.activeDocumentId === docId) {
                clearDocumentFilter();
            }

            await fetchDocuments();
        } catch (err) {
            alert('Failed to delete document: ' + err.message);
        }
    }

    function statCard(label, value) {
        return '<div class="bg-slate-50 rounded-md px-3 py-2.5">' +
                '<div class="text-sm text-slate-400">' + label + '</div>' +
                '<div class="text-base font-semibold text-slate-700">' + escapeHtml(String(value)) + '</div>' +
            '</div>';
    }

    // ── Document search/filter ───────────────────────────────────────────

    docSearch.addEventListener('input', function () {
        clearTimeout(filterTimer);
        filterTimer = setTimeout(function () {
            if (currentView === 'list') {
                renderDocumentList(docSearch.value.trim());
            }
        }, 300);
    });

    // ── Upload UI ────────────────────────────────────────────────────────

    var selectedFile = null;

    uploadBtn.addEventListener('click', function () {
        resetUploadForm();
        uploadModal.showModal();
    });

    uploadModalClose.addEventListener('click', function () {
        uploadModal.close();
    });

    uploadCancelBtn.addEventListener('click', function () {
        uploadModal.close();
    });

    // Close on backdrop click (click on the dialog element itself, not its content)
    uploadModal.addEventListener('click', function (e) {
        if (e.target === uploadModal) {
            uploadModal.close();
        }
    });

    function resetUploadForm() {
        selectedFile = null;
        uploadFileInput.value = '';
        uploadFileName.textContent = '';
        uploadFileName.classList.add('hidden');
        document.getElementById('upload-title').value = '';
        document.getElementById('upload-authors').value = '';
        document.getElementById('upload-year').value = '';
        uploadStatus.innerHTML = '';
        uploadStatus.classList.add('hidden');
        uploadSubmitBtn.disabled = false;
        dropZone.classList.remove('hidden');
    }

    // File input change
    uploadFileInput.addEventListener('change', function () {
        if (this.files && this.files[0]) {
            selectFile(this.files[0]);
        }
    });

    function selectFile(file) {
        var ext = file.name.split('.').pop().toLowerCase();
        if (ext !== 'pdf' && ext !== 'txt') {
            showUploadError('Only PDF and TXT files are accepted.');
            return;
        }
        selectedFile = file;
        uploadFileName.textContent = file.name;
        uploadFileName.classList.remove('hidden');
    }

    // Drag and drop
    var dragCounter = 0;

    dropZone.addEventListener('dragenter', function (e) {
        e.preventDefault();
        dragCounter++;
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragover', function (e) {
        e.preventDefault();
    });

    dropZone.addEventListener('dragleave', function (e) {
        e.preventDefault();
        dragCounter--;
        if (dragCounter === 0) {
            dropZone.classList.remove('drag-over');
        }
    });

    dropZone.addEventListener('drop', function (e) {
        e.preventDefault();
        dragCounter = 0;
        dropZone.classList.remove('drag-over');

        var files = e.dataTransfer.files;
        if (files && files.length) {
            selectFile(files[0]);
        }
    });

    // Prevent browser default file drop behavior on the whole window
    window.addEventListener('dragover', function (e) { e.preventDefault(); });
    window.addEventListener('drop', function (e) { e.preventDefault(); });

    // Submit upload
    uploadSubmitBtn.addEventListener('click', async function () {
        if (!selectedFile) {
            showUploadError('Please select a file.');
            return;
        }

        var title = document.getElementById('upload-title').value.trim();
        var authors = document.getElementById('upload-authors').value.trim();
        var year = document.getElementById('upload-year').value.trim();

        var formData = new FormData();
        formData.append('file', selectedFile);
        if (title) formData.append('title', title);
        if (authors) formData.append('authors', authors);
        if (year) formData.append('pub_year', year);

        uploadSubmitBtn.disabled = true;
        showUploadProgress('Uploading...', 'pending');

        try {
            var resp = await fetch('/api/documents/upload', {
                method: 'POST',
                body: formData,
            });

            if (!resp.ok) {
                var errData = await resp.json().catch(function () { return {}; });
                throw new Error(errData.detail || 'Upload failed');
            }

            var task = await resp.json();
            pollUploadStatus(task.task_id);
        } catch (err) {
            showUploadError(err.message);
            uploadSubmitBtn.disabled = false;
        }
    });

    function pollUploadStatus(taskId) {
        var interval = setInterval(async function () {
            try {
                var resp = await fetch('/api/documents/upload/' + taskId);
                if (!resp.ok) throw new Error('Status check failed');
                var task = await resp.json();

                if (task.status === 'running') {
                    showUploadProgress(task.message || 'Processing...', 'running');
                } else if (task.status === 'completed') {
                    clearInterval(interval);
                    showUploadProgress('Document added successfully!', 'completed');
                    setTimeout(function () {
                        uploadModal.close();
                        fetchDocuments();
                    }, 1500);
                } else if (task.status === 'failed') {
                    clearInterval(interval);
                    showUploadError(task.message || 'Ingestion failed.');
                    uploadSubmitBtn.disabled = false;
                }
            } catch (err) {
                clearInterval(interval);
                showUploadError('Lost connection while checking status.');
                uploadSubmitBtn.disabled = false;
            }
        }, 2000);
    }

    function showUploadProgress(message, status) {
        uploadStatus.classList.remove('hidden');
        var color = status === 'completed' ? 'text-green-600' : 'text-blue-600';
        var icon = status === 'completed'
            ? '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'
            : '<div class="upload-spinner shrink-0"></div>';
        uploadStatus.innerHTML =
            '<div class="flex items-center gap-2 ' + color + '">' +
                icon +
                '<span class="text-sm">' + escapeHtml(message) + '</span>' +
            '</div>';
    }

    function showUploadError(message) {
        uploadStatus.classList.remove('hidden');
        uploadStatus.innerHTML =
            '<p class="text-sm text-red-500">' + escapeHtml(message) + '</p>';
    }

    // ── Utilities ────────────────────────────────────────────────────────

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Init ─────────────────────────────────────────────────────────────
    fetchDocuments();

})();

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const sidebar = document.getElementById('sidebar');
    const menuToggle = document.getElementById('menu-toggle');
    const menuOverlay = document.getElementById('menu-overlay');
    const loader = document.getElementById('loader');
    const noteList = document.getElementById('noteList');
    const searchInput = document.getElementById('searchInput');
    const gameFilter = document.getElementById('gameFilter');
    const icdFilter = document.getElementById('icdFilter');
    const downloadLink = document.getElementById('downloadLink');
    const patientName = document.getElementById('patientName');
    const patientInfo = document.getElementById('patientInfo');
    const placeholder = document.getElementById('placeholder');
    const detailsContainer = document.getElementById('details-container');
    const additionalInfoList = document.getElementById('additional-info-list');

    // Detail fields
    const detailComplaint = document.getElementById('detail-complaint');
    const detailSubjective = document.getElementById('detail-subjective');
    const detailObjective = document.getElementById('detail-objective');
    const detailAssessment = document.getElementById('detail-assessment');
    const detailRecovery = document.getElementById('detail-recovery');

    const csvUrl = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQntM4gqqcetDRtlNI_GPGopx-HfalfC0efux_-KhllcUUStVeeoUYF7SutJdeC-zWbeVn7Vpxkka4R/pub?output=csv'
    let allNotes = [];
    let activeNoteElement = null;

    function toggleMenu() {
        sidebar.classList.toggle('-translate-x-full');
        menuOverlay.classList.toggle('hidden');
    }

    menuToggle.addEventListener('click', toggleMenu);
    menuOverlay.addEventListener('click', toggleMenu);

    /**
     * Custom CSV parser provided by the user.
     * This generator function iterates through a CSV string and yields each row as an array of strings.
     * @param {string} str - The CSV string to parse.
     */
    function* parseCSV(str) {
        // Iterate over each character, keep track of current row and column (of the returned array)
        for (var quote = 0, arr = [], row = 0, col = 0, c = 0; c < str.length; c++) {
            var cc = str[c],
            nc = str[c + 1]; // Current character, next character
            arr[col] = arr[col] || ''; // Create a new column (start with empty string) if necessary

            // If the current character is a quotation mark, and we're inside a
            // quoted field, and the next character is also a quotation mark,
            // add a quotation mark to the current column and skip the next character
            cc == '"' && quote && nc == '"' ? (arr[col] += cc, ++c) :

            // If it's just one quotation mark, begin/end quoted field
            cc == '"' ? (quote = !quote) :

            // If it's a comma and we're not in a quoted field, move on to the next column
            cc == ',' && !quote ? (++col) :

            // If it's a newline (CRLF) and we're not in a quoted field, skip the next character
            // and move on to the next row and move to column 0 of that new row
            cc == '\r' && nc == '\n' && !quote ? (yield arr, col = 0, arr = [], ++c) :

            // If it's a newline (LF or CR) and we're not in a quoted field,
            // move on to the next row and move to column 0 of that new row
            (cc == '\n' || cc == '\r') && !quote ? (yield arr, col = 0, arr = []) :

            // Otherwise, append the current character to the current column
            (arr[col] += cc)
        }
        arr.length && (yield arr)
    }

    /**
     * Fetches data from the CSV URL and processes it using the custom parser.
     */
    async function loadData() {
        try {
            const response = await fetch(csvUrl);
            const csvText = await response.text();

            const parser = parseCSV(csvText);

            // The first row of the CSV is the header
            const headerResult = parser.next();
            if (headerResult.done) {
                throw new Error("CSV file is empty or invalid.");
            }
            const headers = headerResult.value;

            const jsonData = [];
            // Process the rest of the rows
            for (const row of parser) {
                const noteObject = {};
                // Create an object for the current row using the headers as keys
                headers.forEach((header, index) => {
                    noteObject[header] = row[index];
                });
                jsonData.push(noteObject);
            }

            allNotes = jsonData;
            populateFilters(allNotes);
            displayNotes(allNotes);

        } catch (error) {
            console.error("Error loading or parsing CSV data:", error);
            noteList.innerHTML = `<p class="p-4 text-red-600">Failed to load patient data.</p>`;
        } finally {
            loader.style.display = 'none';
        }
    }

    function populateFilters(notes) {
        const games = [...new Set(notes.map(note => note['Game']).filter(Boolean))].sort();

        // Handle comma-separated ICD-10 codes
        const icdCodes = [...new Set(
            notes
            .flatMap(note => note['ICD-10'] ? String(note['ICD-10']).split(',') : [])
            .map(code => code.trim())
            .filter(Boolean)
        )].sort();

        games.forEach(game => gameFilter.add(new Option(game, game)));
        icdCodes.forEach(code => icdFilter.add(new Option(code, code)));
    }

    function displayNotes(notes) {
        noteList.innerHTML = '';
        if (notes.length === 0) {
            noteList.innerHTML = `<p class="p-4 text-gray-500 text-center">No patient notes match.</p>`;
        }

        notes.forEach((note) => {
            const card = document.createElement('div');
            card.className = 'p-4 border-b border-l-4 border-transparent cursor-pointer hover:bg-blue-50 transition-colors duration-150';

            const character = note['Character'] || 'Unknown Patient';
            const complaint = note['Chief Complaint'] || 'No complaint';
            const game = note['Game'] || 'N/A';

            card.innerHTML = `
            <h4 class="font-bold text-blue-800 truncate">${character}</h4>
            <p class="text-sm text-gray-600 truncate">${complaint}</p>
            <p class="text-xs text-gray-400 mt-1">${game}</p>
            `;

            card.addEventListener('click', () => {
                showNoteDetails(note);
                if (activeNoteElement) activeNoteElement.classList.remove('active-note');
                card.classList.add('active-note');
                activeNoteElement = card;
                if (window.innerWidth < 768) toggleMenu();
            });
                noteList.appendChild(card);
        });
    }

    function setAndApplyFilter(filterType, value) {
        if (!value) return;
        searchInput.value = '';
        gameFilter.value = '';
        icdFilter.value = '';

        switch (filterType) {
            case 'character':
                searchInput.value = value;
                break;
            case 'game':
                gameFilter.value = value;
                break;
            case 'icd':
                icdFilter.value = value;
                break;
        }
        applyFilters();

        if (window.innerWidth < 768 && sidebar.classList.contains('-translate-x-full')) {
            toggleMenu();
        }
    }

    function showNoteDetails(note) {
        placeholder.style.display = 'none';
        detailsContainer.style.display = 'block';

        const character = note['Character'] || 'Unknown Patient';
        patientName.textContent = character;
        patientName.onclick = () => setAndApplyFilter('character', character);

        patientInfo.innerHTML = `<span><strong>Age:</strong> ${note['Age'] || 'N/A'}</span>`;

        detailComplaint.textContent = note['Chief Complaint'] || 'N/A';
        detailSubjective.textContent = note['Subjective'] || 'N/A';
        detailObjective.textContent = note['Objective'] || 'N/A';
        detailAssessment.textContent = note['Assessment/ Plan'] || 'N/A';
        detailRecovery.textContent = note['Recovery/ Follow-Up/ Effect of Treatment'] || 'N/A';

        // Create clickable links for Game and each ICD-10 code
        const gameLink = `<a class="filter-link" title="Filter by this game">${note['Game'] || 'N/A'}</a>`;

        const icdCodes = note['ICD-10'] ? String(note['ICD-10']).split(',').map(c => c.trim()) : [];
        const icdLinks = icdCodes.length > 0 ?
        icdCodes.map(code => `<a class="filter-link" title="Filter by this code">${code}</a>`).join(', ') :
        'N/A';

        additionalInfoList.innerHTML = `
        <p><strong>Game:</strong> ${gameLink}</p>
        <p><strong>ICD-10 Code(s):</strong> ${icdLinks}</p>
        <p><strong>Setting:</strong> <span>${note['Setting'] || 'N/A'}</span></p>
        <p><strong>Treatment Accuracy:</strong> <span>${note['Treatment Accuracy'] || 'N/A'}</span></p>
        <p><strong>Recovery Accuracy:</strong> <span>${note['Recovery Accuracy'] || 'N/A'}</span></p>
        <p><strong>Explanation:</strong> <span>${note['Explanation'] || 'N/A'}</span></p>
        `;

        // Add event listeners for the new links
        const gameAnchor = additionalInfoList.querySelector('a[title="Filter by this game"]');
        if (gameAnchor) gameAnchor.onclick = () => setAndApplyFilter('game', note['Game']);

        const icdAnchors = additionalInfoList.querySelectorAll('a[title="Filter by this code"]');
        icdAnchors.forEach(link => {
            link.onclick = () => setAndApplyFilter('icd', link.textContent);
        });
    }

    function applyFilters() {
        const searchQuery = searchInput.value.toLowerCase();
        const selectedGame = gameFilter.value;
        const selectedIcd = icdFilter.value;

        let filteredNotes = allNotes.filter(note => {
            const gameMatch = !selectedGame || note['Game'] === selectedGame;

            // Check if the selected ICD code is in the note's list of codes
            const noteIcdCodes = note['ICD-10'] ? String(note['ICD-10']).split(',').map(c => c.trim()) : [];
            const icdMatch = !selectedIcd || noteIcdCodes.includes(selectedIcd);

            const searchMatch = !searchQuery || Object.values(note).some(value =>
            String(value).toLowerCase().includes(searchQuery)
            );
            return gameMatch && icdMatch && searchMatch;
        });
        displayNotes(filteredNotes);
    }

    searchInput.addEventListener('input', applyFilters);
    gameFilter.addEventListener('change', applyFilters);
    icdFilter.addEventListener('change', applyFilters);

    loadData();
});

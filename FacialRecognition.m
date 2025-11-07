classdef FacialRecognition < matlab.apps.AppBase
    % =====================================================================
    % UI PROPERTIES
    % =====================================================================
    properties (Access = public)
        UIFigure matlab.ui.Figure
        MainPanel matlab.ui.container.Panel
        HeaderPanel matlab.ui.container.Panel
        ControlPanel matlab.ui.container.Panel
        DisplayPanel matlab.ui.container.Panel
        InfoPanel matlab.ui.container.Panel
        TitleLabel matlab.ui.control.Label
        SubtitleLabel matlab.ui.control.Label
        RegisterBtn matlab.ui.control.Button
        ScanBtn matlab.ui.control.Button
        IdentifyBtn matlab.ui.control.Button
        ViewDatabaseBtn matlab.ui.control.Button   % now used as "Refresh"
        ScanFPBtn matlab.ui.control.Button
        FusionIdentifyBtn matlab.ui.control.Button
        DeleteBtn matlab.ui.control.Button

        % face axes
        ScanImageAxes matlab.ui.control.UIAxes
        FoundImageAxes matlab.ui.control.UIAxes

        % bottom info
        StatusLabel matlab.ui.control.Label
        MatchLabel matlab.ui.control.Label
        ConfidenceLabel matlab.ui.control.Label
        DatabaseCountLabel matlab.ui.control.Label

        % database list
        DBList matlab.ui.control.ListBox

        % fingerprint panel
        FPPanel matlab.ui.container.Panel
        FPScanAxes matlab.ui.control.UIAxes
        FPMatchAxes matlab.ui.control.UIAxes
    end

    % =====================================================================
    % INTERNAL DATA
    % =====================================================================
    properties (Access = private)
        faceDetector
        userFaces = {}           % stored display faces (80x80)
        userFaceFeatures = {}    % deep features
        userNames = {}           % names aligned with faces
        userFingerprints = {}    % skeleton FP aligned with names
        scanImage                % scanned face (display)
        scanFeatures             % scanned face features
        scannedFingerprintSkel   % scanned FP skeleton (for fusion)
        scannedFingerprintImg    % original FP scan (optional)
        dataFile = 'identityData.mat'
        currentViewIndex = 1
        featureNet               % alexnet if available
    end

    % =====================================================================
    % PRIVATE METHODS
    % =====================================================================
    methods (Access = private)

        % ================================================================
        % DEEP FEATURE EXTRACTION (kept)
        % ================================================================
        function features = extractDeepFeatures(app, face)
            try
                faceResized = imresize(face, [224 224]);
                if size(faceResized, 3) == 1
                    faceResized = cat(3, faceResized, faceResized, faceResized);
                end
                features = activations(app.featureNet, faceResized, 'fc7', 'OutputAs', 'rows');
                features = features / norm(features);
            catch
                features = app.extractRobustFeatures(face);
            end
        end

        function features = extractRobustFeatures(~, face)
            if size(face, 3) == 3
                faceGray = rgb2gray(face);
            else
                faceGray = face;
            end
            faceGray = imresize(faceGray, [80 80]);

            % HOG
            try
                hogFeatures = extractHOGFeatures(faceGray, 'CellSize', [8 8]);
            catch
                hogFeatures = [];
            end

            % LBP
            try
                lbpFeatures = extractLBPFeatures(faceGray, 'Upright', false);
            catch
                lbpFeatures = [];
            end

            % SURF aggregated
            try
                points = detectSURFFeatures(faceGray, 'MetricThreshold', 100);
                [surfFeatures, ~] = extractFeatures(faceGray, points);
                if ~isempty(surfFeatures)
                    surfFeatures = mean(surfFeatures, 1);
                else
                    surfFeatures = zeros(1, 64);
                end
            catch
                surfFeatures = zeros(1, 64);
            end

            % block hist
            [h, w] = size(faceGray);
            gridSize = 4;
            cellH = floor(h / gridSize);
            cellW = floor(w / gridSize);
            histFeatures = [];
            for i = 1:gridSize
                for j = 1:gridSize
                    rs = (i-1)*cellH + 1; re = min(i*cellH, h);
                    cs = (j-1)*cellW + 1; ce = min(j*cellW, w);
                    cellImg = faceGray(rs:re, cs:ce);
                    hh = imhist(cellImg);
                    hh = hh / sum(hh);
                    histFeatures = [histFeatures; hh];
                end
            end

            features = [hogFeatures, lbpFeatures, surfFeatures, histFeatures];

            if ~isempty(features)
                features = features / (norm(features) + eps);
            else
                features = zeros(1, 1000);
            end
        end

        function similarity = compareFeatures(~, f1, f2)
            if isempty(f1) || isempty(f2)
                similarity = 0;
                return;
            end
            m = min(numel(f1), numel(f2));
            f1 = f1(1:m); f2 = f2(1:m);

            cosSim = dot(f1, f2) / (norm(f1)*norm(f2) + eps);
            cosSim = max(0, cosSim);

            eud = norm(f1 - f2);
            eSim = 1 / (1 + eud);

            m1 = mean(f1); m2 = mean(f2);
            num = sum((f1-m1).*(f2-m2));
            den = sqrt(sum((f1-m1).^2) * sum((f2-m2).^2));
            if den > eps
                corrSim = num / den;
                corrSim = (corrSim + 1)/2;
            else
                corrSim = 0;
            end

            similarity = 0.7*cosSim + 0.2*eSim + 0.1*corrSim;
        end

        % ================================================================
        % FACE DETECTION + QUALITY (kept)
        % ================================================================
        function [face, isValid, qualityReport] = detectAndCropFaceEnhanced(app, img, detector)
            face = [];
            isValid = false;
            qualityReport = struct();
            qualityReport.qualityScore = 0;
            qualityReport.qualityGrade = 'Very Poor';
            qualityReport.error = '';

            if size(img,3) == 1
                imgRGB = cat(3, img, img, img);
            else
                imgRGB = img;
            end

            if size(img,3) == 3
                grayImg = rgb2gray(img);
            else
                grayImg = img;
            end

            bbox = step(detector, grayImg);
            if isempty(bbox)
                qualityReport.error = 'No face detected in image';
                qualityReport.faceCount = 0;
                return;
            end

            numFaces = size(bbox,1);
            qualityReport.faceCount = numFaces;
            if numFaces > 1
                areas = bbox(:,3).*bbox(:,4);
                [~, idx] = max(areas);
                bbox = bbox(idx,:);
                qualityReport.warning = sprintf('%d faces detected, using largest', numFaces);
            end

            faceWidth = bbox(3); faceHeight = bbox(4);
            if faceWidth < 50 || faceHeight < 50
                qualityReport.error = sprintf('Face too small (%dx%d)', round(faceWidth), round(faceHeight));
                return;
            end

            pad = 0.2;
            x = max(1, bbox(1) - bbox(3)*pad);
            y = max(1, bbox(2) - bbox(4)*pad);
            w = min(size(img,2)-x, bbox(3)*(1+2*pad));
            h = min(size(img,1)-y, bbox(4)*(1+2*pad));
            face = imcrop(imgRGB, [x y w h]);

            blurScore = app.estimateBlur(face);
            grayFace = rgb2gray(face);
            brightness = mean(grayFace(:));
            contrast = std(double(grayFace(:)));

            [fh, fw, ~] = size(face);
            if fh < 80 || fw < 80
                qualityReport.error = 'Face resolution too low for reliable matching';
                return;
            end

            try
                eyeDetector = vision.CascadeObjectDetector('EyePairBig');
                eyeBbox = step(eyeDetector, grayFace);
                eyesDetected = ~isempty(eyeBbox);
            catch
                eyesDetected = false;
            end

            score = 0;
            if blurScore >= 200
                score = score + 30;
            elseif blurScore >= 100
                score = score + 20;
            else
                score = score + 10;
            end

            if brightness >= 80 && brightness <= 180
                score = score + 25;
            elseif brightness >= 60 && brightness <= 200
                score = score + 15;
            else
                score = score + 5;
            end

            if contrast >= 50
                score = score + 20;
            elseif contrast >= 30
                score = score + 15;
            else
                score = score + 5;
            end

            if fh >= 150 && fw >= 150
                score = score + 15;
            elseif fh >= 100 && fw >= 100
                score = score + 10;
            else
                score = score + 5;
            end

            if eyesDetected
                score = score + 10;
            end

            qualityReport.blurScore = blurScore;
            qualityReport.brightness = brightness;
            qualityReport.contrast = contrast;
            qualityReport.eyesDetected = eyesDetected;
            qualityReport.qualityScore = score;
            qualityReport.qualityGrade = app.getQualityGrade(score);

            if score >= 50
                isValid = true;
                qualityReport.status = 'PASS';
            else
                isValid = false;
                qualityReport.status = 'FAIL';
                qualityReport.error = sprintf('Quality too low (Score: %d/100)', score);
            end
        end

        % ================================================================
        % SAVE / LOAD (extended to include fingerprints)
        % ================================================================
        function saveData(app)
            userFaces = app.userFaces;
            userNames = app.userNames;
            userFaceFeatures = app.userFaceFeatures;
            userFingerprints = app.userFingerprints;
            save(app.dataFile, 'userFaces', 'userNames', 'userFaceFeatures', 'userFingerprints');
        end

        function loadData(app)
            if isfile(app.dataFile)
                data = load(app.dataFile);
                if isfield(data, 'userFaces'), app.userFaces = data.userFaces; end
                if isfield(data, 'userNames'), app.userNames = data.userNames; end
                if isfield(data, 'userFaceFeatures'), app.userFaceFeatures = data.userFaceFeatures; end
                if isfield(data, 'userFingerprints')
                    app.userFingerprints = data.userFingerprints;
                else
                    % align sizes if old mat
                    app.userFingerprints = cell(1, numel(app.userNames));
                end
            end
        end

        function updateDatabaseCount(app)
            count = length(app.userNames);
            app.DatabaseCountLabel.Text = sprintf('Database: %d identities', count);
        end

        function refreshDatabaseList(app)
            % refresh listbox items to match userNames
            if isempty(app.userNames)
                app.DBList.Items = {};
            else
                app.DBList.Items = app.userNames;
            end
        end

        % ================================================================
        % REGISTRATION (face + fp + name)
        % ================================================================
        function registerNewIdentity(app)
            % 1) pick face
            [file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Register a new identity (FACE)');
            if isequal(file, 0)
                return;
            end

            try
                img = imread(fullfile(path, file));
                [face, isValid, qualityReport] = app.detectAndCropFaceEnhanced(img, app.faceDetector);

                if ~isValid
                    msg = sprintf('Registration failed:\n%s\nQuality: %d/100 (%s)', ...
                        qualityReport.error, qualityReport.qualityScore, qualityReport.qualityGrade);
                    uialert(app.UIFigure, msg, 'Quality Check Failed', 'Icon', 'warning');
                    app.StatusLabel.Text = 'Registration failed - Image quality insufficient';
                    app.StatusLabel.FontColor = [1 0.4 0.4];
                    return;
                end

                if qualityReport.qualityScore < 70
                    warn = sprintf('Image quality: %s (%d/100)\nContinue?', ...
                        qualityReport.qualityGrade, qualityReport.qualityScore);
                    choice = uiconfirm(app.UIFigure, warn, 'Quality Warning', ...
                        'Options', {'Register Anyway', 'Cancel'}, ...
                        'DefaultOption', 1, 'Icon', 'warning');
                    if strcmp(choice, 'Cancel')
                        app.StatusLabel.Text = 'Registration cancelled by user';
                        app.StatusLabel.FontColor = [1 0.7 0.3];
                        return;
                    end
                end

                % ask name
                answer = inputdlg({'Enter name for this person:'}, 'Identity Name', [1 50], {''});
                if isempty(answer)
                    app.StatusLabel.Text = 'Registration cancelled (no name).';
                    app.StatusLabel.FontColor = [1 0.7 0.3];
                    return;
                end
                name = strtrim(answer{1});
                if isempty(name)
                    app.StatusLabel.Text = 'Registration cancelled (empty name).';
                    app.StatusLabel.FontColor = [1 0.7 0.3];
                    return;
                end

                % extract face features
                faceFeatures = app.extractDeepFeatures(face);

                % preprocess for display
                if size(face,3) == 3
                    faceGray = rgb2gray(face);
                else
                    faceGray = face;
                end
                faceEq = adapthisteq(faceGray, 'ClipLimit', 0.02);
                faceEq = wiener2(faceEq, [3 3]);
                faceResized = imresize(faceEq, [80 80]);

                % append to db
                app.userFaces{end+1} = faceResized;
                app.userFaceFeatures{end+1} = faceFeatures;
                app.userNames{end+1} = name;

                % 2) ask for fingerprint (optional but recommended)
                [fpf, fpp] = uigetfile({'*.jpg;*.png;*.bmp'}, ...
                    sprintf('Select fingerprint for %s (optional)', name));
                if isequal(fpf,0)
                    app.userFingerprints{end+1} = [];
                else
                    fpImg = imread(fullfile(fpp,fpf));
                    [~,~,skel,~] = fpEnhance(fpImg);
                    app.userFingerprints{end+1} = skel;
                    imshow(skel, 'Parent', app.FPScanAxes);
                    title(app.FPScanAxes, sprintf('FP of %s', name), 'Color', [0.4 1 0.6]);
                end

                app.saveData();
                app.updateDatabaseCount();
                app.refreshDatabaseList();

                app.StatusLabel.Text = sprintf('✓ Identity registered: %s', name);
                app.StatusLabel.FontColor = [0.4 1 0.6];

            catch ME
                app.StatusLabel.Text = sprintf('Registration failed: %s', ME.message);
                app.StatusLabel.FontColor = [1 0.4 0.4];
            end
        end

        % ================================================================
        % SCAN FACE (same)
        % ================================================================
        function scanIdentity(app)
            [file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Scan identity photo');
            if isequal(file, 0)
                return;
            end
            try
                img = imread(fullfile(path, file));
                [face, isValid, qualityReport] = app.detectAndCropFaceEnhanced(img, app.faceDetector);

                if ~isValid
                    errorMsg = sprintf('Scan failed:\n%s\nQuality: %d/100 (%s)', ...
                        qualityReport.error, qualityReport.qualityScore, qualityReport.qualityGrade);
                    uialert(app.UIFigure, errorMsg, 'Quality Check Failed', 'Icon', 'warning');
                    app.StatusLabel.Text = 'Scan failed - Image quality insufficient';
                    app.StatusLabel.FontColor = [1 0.4 0.4];
                    return;
                end

                if qualityReport.qualityScore < 70
                    warningMsg = sprintf('Image quality: %s (%d/100)\nProceed?', ...
                        qualityReport.qualityGrade, qualityReport.qualityScore);
                    choice = uiconfirm(app.UIFigure, warningMsg, 'Quality Warning', ...
                        'Options', {'Continue', 'Cancel'}, 'DefaultOption', 1, 'Icon', 'warning');
                    if strcmp(choice, 'Cancel')
                        app.StatusLabel.Text = 'Scan cancelled by user';
                        app.StatusLabel.FontColor = [1 0.7 0.3];
                        return;
                    end
                end

                app.scanFeatures = app.extractDeepFeatures(face);

                if size(face,3) == 3
                    faceGray = rgb2gray(face);
                else
                    faceGray = face;
                end
                faceEq = adapthisteq(faceGray, 'ClipLimit', 0.02);
                faceEq = wiener2(faceEq, [3 3]);
                app.scanImage = imresize(faceEq, [80 80]);

                imshow(face, 'Parent', app.ScanImageAxes);
                app.ScanImageAxes.XColor = [0.3 0.8 0.5];
                app.ScanImageAxes.YColor = [0.3 0.8 0.5];
                app.ScanImageAxes.LineWidth = 3;
                title(app.ScanImageAxes, sprintf('Scanned Photo (Quality: %s)', qualityReport.qualityGrade), ...
                    'FontSize', 14, 'Color', [0.4 1 0.6]);
                app.StatusLabel.Text = sprintf('Photo scanned - Quality: %s. Ready to identify.', qualityReport.qualityGrade);
                app.StatusLabel.FontColor = [0.4 1 0.6];

                app.MatchLabel.Text = '';
                app.ConfidenceLabel.Text = '';
                cla(app.FoundImageAxes);
                title(app.FoundImageAxes, 'Match result', 'FontSize', 14, 'Color', [0.7 0.7 0.7]);

            catch ME
                app.StatusLabel.Text = sprintf('Scan failed: %s', ME.message);
                app.StatusLabel.FontColor = [1 0.4 0.4];
            end
        end

        % ================================================================
        % FACE-ONLY IDENTIFICATION (original)
        % ================================================================
        function performIdentification(app)
            if isempty(app.scanImage) || isempty(app.scanFeatures)
                app.StatusLabel.Text = 'Please scan a photo first';
                app.StatusLabel.FontColor = [1 0.7 0.3];
                return;
            end

            if isempty(app.userFaces)
                app.StatusLabel.Text = 'No match detected. Database is empty';
                app.StatusLabel.FontColor = [1 0.4 0.4];
                app.MatchLabel.Text = 'NO MATCH DETECTED';
                app.MatchLabel.FontColor = [1 0.4 0.4];
                app.ConfidenceLabel.Text = 'Confidence: 0.0%';
                cla(app.FoundImageAxes);
                title(app.FoundImageAxes, 'Match Result - Empty', 'FontSize', 14, 'Color', [1 0.4 0.4]);
                app.FoundImageAxes.XColor = [1 0.4 0.4];
                app.FoundImageAxes.YColor = [1 0.4 0.4];
                app.FoundImageAxes.LineWidth = 3;
                return;
            end

            app.StatusLabel.Text = 'Processing identification using deep features...';
            app.StatusLabel.FontColor = [0.4 0.7 1];
            drawnow;

            similarities = zeros(1, length(app.userFaceFeatures));
            for i = 1:length(app.userFaceFeatures)
                similarities(i) = app.compareFeatures(app.scanFeatures, app.userFaceFeatures{i});
            end

            [maxSimilarity, bestIdx] = max(similarities);
            ABSOLUTE_MIN_SIMILARITY = 0.40;

            if maxSimilarity < ABSOLUTE_MIN_SIMILARITY
                app.MatchLabel.Text = 'NO MATCH DETECTED';
                app.MatchLabel.FontColor = [1 0.4 0.4];
                app.StatusLabel.Text = sprintf('Unknown person (Best match only %.1f%% - too low)', maxSimilarity*100);
                app.StatusLabel.FontColor = [1 0.4 0.4];

                cla(app.FoundImageAxes);
                title(app.FoundImageAxes, 'Unknown Person', 'FontSize', 14, 'Color', [1 0.4 0.4]);
                app.FoundImageAxes.XColor = [1 0.4 0.4];
                app.FoundImageAxes.YColor = [1 0.4 0.4];
                app.FoundImageAxes.LineWidth = 3;

                app.ConfidenceLabel.Text = sprintf('All matches below %.0f%% threshold', ABSOLUTE_MIN_SIMILARITY * 100);
                return;
            end

            confidence = maxSimilarity * 100;

            sortedSim = sort(similarities, 'descend');
            discrimination = 0;
            if numel(sortedSim) > 1
                secondBest = sortedSim(2);
                discrimination = (maxSimilarity - secondBest) / (maxSimilarity + eps) * 100;
                if discrimination > 15 && maxSimilarity > 0.50
                    confidence = min(100, confidence * 1.10);
                end
            end

            baseThreshold = 60;
            if length(app.userFaces) > 10
                adaptiveThreshold = baseThreshold + 5;
            elseif length(app.userFaces) > 5
                adaptiveThreshold = baseThreshold + 2;
            else
                adaptiveThreshold = baseThreshold;
            end

            simStd = std(similarities);
            if simStd < 0.05
                adaptiveThreshold = adaptiveThreshold + 10;
            elseif simStd < 0.10
                adaptiveThreshold = adaptiveThreshold + 5;
            end

            if discrimination < 10 && length(sortedSim) > 1
                adaptiveThreshold = adaptiveThreshold + 8;
                app.StatusLabel.Text = 'Warning: Multiple similar matches detected';
            end

            if confidence >= adaptiveThreshold
                app.MatchLabel.Text = sprintf('IDENTIFIED: %s', app.userNames{bestIdx});
                app.MatchLabel.FontColor = [0.4 1 0.6];

                if confidence >= 85
                    app.StatusLabel.Text = 'Strong match - Very high confidence';
                    app.StatusLabel.FontColor = [0.4 1 0.6];
                elseif confidence >= 70
                    app.StatusLabel.Text = 'Good match - High confidence';
                    app.StatusLabel.FontColor = [0.4 1 0.6];
                elseif confidence >= 60
                    app.StatusLabel.Text = 'Possible match - Moderate confidence (verify recommended)';
                    app.StatusLabel.FontColor = [1 0.9 0.3];
                else
                    app.StatusLabel.Text = 'Weak match - Low confidence (manual verification required)';
                    app.StatusLabel.FontColor = [1 0.7 0.3];
                end

                borderColor = [0.3 0.8 0.5];
                imshow(app.userFaces{bestIdx}, 'Parent', app.FoundImageAxes);
                app.FoundImageAxes.XColor = borderColor;
                app.FoundImageAxes.YColor = borderColor;
                app.FoundImageAxes.LineWidth = 3;
                title(app.FoundImageAxes, sprintf('Found: %s', app.userNames{bestIdx}), ...
                    'FontSize', 14, 'Color', borderColor);
            else
                app.MatchLabel.Text = 'NO MATCH DETECTED';
                app.MatchLabel.FontColor = [1 0.4 0.4];

                if confidence < ABSOLUTE_MIN_SIMILARITY * 100
                    app.StatusLabel.Text = sprintf('Unknown person (Best: %.1f%%)', confidence);
                else
                    app.StatusLabel.Text = sprintf('No confident match (Best: %.1f%% < %.1f%% threshold)', ...
                        confidence, adaptiveThreshold);
                end
                app.StatusLabel.FontColor = [1 0.4 0.4];

                cla(app.FoundImageAxes);
                title(app.FoundImageAxes, 'Match Result - No Match', 'FontSize', 14, 'Color', [1 0.4 0.4]);
                app.FoundImageAxes.XColor = [1 0.4 0.4];
                app.FoundImageAxes.YColor = [1 0.4 0.4];
                app.FoundImageAxes.LineWidth = 3;
            end

            if length(similarities) > 1
                [sortedScores, sortedIdx] = sort(similarities, 'descend');
                dbg = 'Top matches: ';
                for k = 1:min(3, length(similarities))
                    dbg = sprintf('%s%s(%.1f%%) ', dbg, app.userNames{sortedIdx(k)}, sortedScores(k)*100);
                end
                app.ConfidenceLabel.Text = sprintf('Confidence: %.1f%% | Th: %.1f%% | %s', ...
                    confidence, adaptiveThreshold, dbg);
            else
                app.ConfidenceLabel.Text = sprintf('Confidence: %.1f%% | Th: %.1f%% | Method: Deep Features', ...
                    confidence, adaptiveThreshold);
            end
        end

        % ================================================================
        % VIEW BY LISTBOX (on selection change)
        % ================================================================
        function DBListValueChanged(app, ~)
            idx = app.DBList.Value;
            if ischar(idx)
                % happens when items empty
                return;
            end
        end

        % we will handle selection using event.Value
        function DBListValueChanged2(app, event)
            % event.Value is string (name)
            name = event.Value;
            if isempty(name)
                return;
            end
            % find index
            idx = find(strcmp(app.userNames, name), 1);
            if isempty(idx), return; end

            % show face
            if idx <= numel(app.userFaces) && ~isempty(app.userFaces{idx})
                imshow(app.userFaces{idx}, 'Parent', app.FoundImageAxes);
                app.FoundImageAxes.XColor = [0.4 0.7 1];
                app.FoundImageAxes.YColor = [0.4 0.7 1];
                app.FoundImageAxes.LineWidth = 3;
                title(app.FoundImageAxes, sprintf('DB: %s', name), 'Color', [0.4 0.7 1]);
            else
                cla(app.FoundImageAxes);
                title(app.FoundImageAxes, 'No face stored', 'Color', [1 0.4 0.4]);
            end

            % show fingerprint if exists
            if idx <= numel(app.userFingerprints) && ~isempty(app.userFingerprints{idx})
                imshow(app.userFingerprints{idx}, 'Parent', app.FPMatchAxes);
                title(app.FPMatchAxes, 'Stored Fingerprint', 'Color', [0.4 1 0.6]);
            else
                cla(app.FPMatchAxes);
                title(app.FPMatchAxes, 'No fingerprint stored', 'Color', [1 0.4 0.4]);
            end

            app.StatusLabel.Text = sprintf('Viewing: %s', name);
            app.StatusLabel.FontColor = [0.4 0.7 1];
        end

        % ================================================================
        % VIEW DATABASE BUTTON (now = refresh list)
        % ================================================================
        function viewDatabase(app)
            app.refreshDatabaseList();
            app.StatusLabel.Text = 'Database list refreshed.';
            app.StatusLabel.FontColor = [0.4 0.7 1];
        end

        % ================================================================
        % SCAN FINGERPRINT
        % ================================================================
        function scanFingerprint(app)
            [f,p] = uigetfile({'*.jpg;*.png;*.bmp'}, 'Scan fingerprint for identification');
            if isequal(f,0), return; end
            img = imread(fullfile(p,f));
            [~,~,skel,~] = fpEnhance(img);
            app.scannedFingerprintSkel = skel;
            app.scannedFingerprintImg = img;

            imshow(skel, 'Parent', app.FPScanAxes);
            title(app.FPScanAxes, 'Scanned Fingerprint', 'Color', [0.4 1 0.6]);

            app.StatusLabel.Text = 'Fingerprint scanned. You can now press Identify (Fusion).';
            app.StatusLabel.FontColor = [0.4 1 0.6];
        end

        % ================================================================
        % FUSION IDENTIFICATION
        % ================================================================
        function performFusionIdentification(app)
            if isempty(app.scanFeatures)
                app.StatusLabel.Text = 'Scan FACE first.';
                app.StatusLabel.FontColor = [1 0.6 0.2];
                return;
            end
            if isempty(app.scannedFingerprintSkel)
                app.StatusLabel.Text = 'Scan FINGERPRINT first.';
                app.StatusLabel.FontColor = [1 0.6 0.2];
                return;
            end
            if isempty(app.userNames)
                app.StatusLabel.Text = 'Database empty.';
                app.StatusLabel.FontColor = [1 0.4 0.4];
                return;
            end

            n = numel(app.userNames);
            faceScores = zeros(1,n);
            fpScores   = zeros(1,n);
            fused      = zeros(1,n);

            for i = 1:n
                faceScores(i) = app.compareFeatures(app.scanFeatures, app.userFaceFeatures{i});

                if i <= numel(app.userFingerprints) && ~isempty(app.userFingerprints{i})
                    fpScores(i) = fpMatch(app.scannedFingerprintSkel, app.userFingerprints{i});
                else
                    fpScores(i) = 0;
                end

                fused(i) = 0.6*faceScores(i) + 0.4*fpScores(i);
            end

            [bestScore, bestIdx] = max(fused);
            name = app.userNames{bestIdx};

            if bestScore >= 0.55
                app.MatchLabel.Text = sprintf('FUSION: %s', name);
                app.MatchLabel.FontColor = [0.4 1 0.6];
                imshow(app.userFaces{bestIdx}, 'Parent', app.FoundImageAxes);
                app.StatusLabel.Text = sprintf('Fusion success: %s | Face: %.1f%% | FP: %.1f%% | Final: %.1f%%', ...
                    name, faceScores(bestIdx)*100, fpScores(bestIdx)*100, bestScore*100);
                app.StatusLabel.FontColor = [0.4 1 0.6];

                if bestIdx <= numel(app.userFingerprints) && ~isempty(app.userFingerprints{bestIdx})
                    imshow(app.userFingerprints{bestIdx}, 'Parent', app.FPMatchAxes);
                    title(app.FPMatchAxes, 'Matched Fingerprint', 'Color', [0.4 1 0.6]);
                end
            else
                app.MatchLabel.Text = 'FUSION: NO MATCH';
                app.MatchLabel.FontColor = [1 0.4 0.4];
                app.StatusLabel.Text = sprintf('Fusion failed. Best: %.1f%% (need > 55%%)', bestScore*100);
                app.StatusLabel.FontColor = [1 0.4 0.4];
            end
        end

        % ================================================================
        % DELETE IDENTITY
        % ================================================================
        function DeleteBtnPushed(app, ~)
            if isempty(app.userNames)
                uialert(app.UIFigure, 'Database is empty.', 'Delete Failed', 'Icon', 'warning');
                return;
            end

            % Try to use currently selected item in listbox
            selectedName = '';
            if ~isempty(app.DBList.Items) && ~isempty(app.DBList.Value)
                selectedName = app.DBList.Value;
            end

            if isempty(selectedName)
                % fallback: list dialog
                [idx, tf] = listdlg('PromptString', 'Select identity to delete:', ...
                    'SelectionMode', 'single', 'ListString', app.userNames, ...
                    'Name', 'Delete Identity');
                if ~tf
                    app.StatusLabel.Text = 'Deletion cancelled.';
                    app.StatusLabel.FontColor = [1 0.7 0.3];
                    return;
                end
                selectedName = app.userNames{idx};
            end

            idx = find(strcmp(app.userNames, selectedName), 1);
            if isempty(idx)
                app.StatusLabel.Text = 'Deletion cancelled (not found).';
                app.StatusLabel.FontColor = [1 0.7 0.3];
                return;
            end

            choice = uiconfirm(app.UIFigure, ...
                sprintf('Delete "%s" from database?', selectedName), ...
                'Confirm Deletion', ...
                'Options', {'Yes, Delete', 'Cancel'}, ...
                'DefaultOption', 2, 'Icon', 'warning');

            if ~strcmp(choice, 'Yes, Delete')
                app.StatusLabel.Text = 'Deletion cancelled.';
                app.StatusLabel.FontColor = [1 0.7 0.3];
                return;
            end

            % delete aligned entries
            app.userNames(idx) = [];
            if idx <= numel(app.userFaces), app.userFaces(idx) = []; end
            if idx <= numel(app.userFaceFeatures), app.userFaceFeatures(idx) = []; end
            if idx <= numel(app.userFingerprints), app.userFingerprints(idx) = []; end

            app.saveData();
            app.updateDatabaseCount();
            app.refreshDatabaseList();

            cla(app.FoundImageAxes);
            cla(app.FPMatchAxes);

            app.StatusLabel.Text = sprintf('🗑️ Deleted identity: %s', selectedName);
            app.StatusLabel.FontColor = [1 0.4 0.4];
        end

        % ================================================================
        % UTILITIES
        % ================================================================
        function blurScore = estimateBlur(~, img)
            if size(img,3) == 3
                img = rgb2gray(img);
            end
            laplacian = fspecial('laplacian');
            imgLap = imfilter(img, laplacian, 'replicate');
            blurScore = std2(imgLap)^2;
        end

        function grade = getQualityGrade(~, score)
            if score >= 85
                grade = 'Excellent';
            elseif score >= 70
                grade = 'Good';
            elseif score >= 60
                grade = 'Acceptable';
            elseif score >= 40
                grade = 'Poor';
            else
                grade = 'Very Poor';
            end
        end

        function advice = getQualityAdvice(~, report)
            advice = '';
            if report.blurScore < 100
                advice = [advice, '• Use a sharper, focused image', newline];
            end
            if report.brightness < 40
                advice = [advice, '• Image is too dark, use better lighting', newline];
            elseif report.brightness > 220
                advice = [advice, '• Image is overexposed, reduce lighting', newline];
            end
            if report.contrast < 30
                advice = [advice, '• Low contrast, ensure good lighting conditions', newline];
            end
            if isfield(report, 'eyesDetected') && ~report.eyesDetected
                advice = [advice, '• Ensure face is clearly visible and frontal', newline];
            end
            if isempty(advice)
                advice = 'Image meets minimum requirements but could be better';
            end
        end

    end

    % =====================================================================
    % CALLBACK REGISTRATION
    % =====================================================================
    methods (Access = private)
        function startupFcn(app)
            app.faceDetector = vision.CascadeObjectDetector();

            try
                app.featureNet = alexnet;
                app.StatusLabel.Text = 'System ready - Using AlexNet for deep features';
            catch
                app.featureNet = [];
                app.StatusLabel.Text = 'System ready - Using traditional features (AlexNet not available)';
            end

            app.loadData();
            app.updateDatabaseCount();
            app.refreshDatabaseList();
            app.StatusLabel.FontColor = [0.4 0.7 1];
        end

        function RegisterBtnPushed(app, ~)
            app.registerNewIdentity();
        end

        function ScanBtnPushed(app, ~)
            app.scanIdentity();
        end

        function IdentifyBtnPushed(app, ~)
            app.performIdentification();
        end

        function ViewDatabaseBtnPushed(app, ~)
            app.viewDatabase();
        end

        function ScanFPBtnPushed(app, ~)
            app.scanFingerprint();
        end

        function FusionIdentifyBtnPushed(app, ~)
            app.performFusionIdentification();
        end

        function DBListValueChangedWrapper(app, event)
            app.DBListValueChanged2(event);
        end
    end

    % =====================================================================
    % UI CREATION
    % =====================================================================
    methods (Access = private)
        function createComponents(app)
            % colors
            darkBg = [0.12 0.12 0.15];
            darkerBg = [0.08 0.08 0.10];
            panelBg = [0.15 0.15 0.18];
            headerBg = [0.10 0.10 0.13];
            accentCyan = [0.3 0.8 0.9];
            textColor = [0.9 0.9 0.9];

            % figure (slightly taller)
            app.UIFigure = uifigure('Position', [100 100 1000 750]);
            app.UIFigure.Name = 'Facial + Fingerprint Recognition System';
            app.UIFigure.Color = darkBg;

            % main
            app.MainPanel = uipanel(app.UIFigure);
            app.MainPanel.Position = [15 15 970 720];
            app.MainPanel.BackgroundColor = darkerBg;
            app.MainPanel.BorderType = 'none';

            % header
            app.HeaderPanel = uipanel(app.MainPanel);
            app.HeaderPanel.Position = [20 620 930 80];
            app.HeaderPanel.BackgroundColor = headerBg;
            app.HeaderPanel.BorderType = 'none';

            app.TitleLabel = uilabel(app.HeaderPanel);
            app.TitleLabel.Position = [30 35 870 30];
            app.TitleLabel.Text = 'FACIAL + FINGERPRINT RECOGNITION SYSTEM';
            app.TitleLabel.FontSize = 22;
            app.TitleLabel.FontWeight = 'bold';
            app.TitleLabel.FontColor = accentCyan;
            app.TitleLabel.HorizontalAlignment = 'center';

            app.SubtitleLabel = uilabel(app.HeaderPanel);
            app.SubtitleLabel.Position = [30 10 870 20];
            app.SubtitleLabel.Text = 'Multimodal biometric with fusion • Face pipeline preserved';
            app.SubtitleLabel.FontSize = 12;
            app.SubtitleLabel.FontColor = [0.6 0.6 0.65];
            app.SubtitleLabel.HorizontalAlignment = 'center';

            % control panel (buttons)
            app.ControlPanel = uipanel(app.MainPanel);
            app.ControlPanel.Position = [20 550 930 60];
            app.ControlPanel.BackgroundColor = panelBg;
            app.ControlPanel.BorderType = 'line';
            app.ControlPanel.BorderWidth = 1;
            app.ControlPanel.HighlightColor = [0.25 0.25 0.28];

            % buttons
            app.RegisterBtn = uibutton(app.ControlPanel, 'push');
            app.RegisterBtn.Position = [20 15 130 30];
            app.RegisterBtn.Text = 'Register Identity';
            app.RegisterBtn.FontWeight = 'bold';
            app.RegisterBtn.BackgroundColor = [0 0 0];
            app.RegisterBtn.FontColor = [1 1 1];
            app.RegisterBtn.ButtonPushedFcn = createCallbackFcn(app, @RegisterBtnPushed, true);

            app.ScanBtn = uibutton(app.ControlPanel, 'push');
            app.ScanBtn.Position = [160 15 110 30];
            app.ScanBtn.Text = 'Scan Face';
            app.ScanBtn.FontWeight = 'bold';
            app.ScanBtn.BackgroundColor = [0 0 0];
            app.ScanBtn.FontColor = [1 1 1];
            app.ScanBtn.ButtonPushedFcn = createCallbackFcn(app, @ScanBtnPushed, true);

            app.IdentifyBtn = uibutton(app.ControlPanel, 'push');
            app.IdentifyBtn.Position = [280 15 120 30];
            app.IdentifyBtn.Text = 'Identify (Face)';
            app.IdentifyBtn.FontWeight = 'bold';
            app.IdentifyBtn.BackgroundColor = [0 0 0];
            app.IdentifyBtn.FontColor = [1 1 1];
            app.IdentifyBtn.ButtonPushedFcn = createCallbackFcn(app, @IdentifyBtnPushed, true);

            app.ViewDatabaseBtn = uibutton(app.ControlPanel, 'push');
            app.ViewDatabaseBtn.Position = [410 15 110 30];
            app.ViewDatabaseBtn.Text = 'Refresh DB';
            app.ViewDatabaseBtn.FontWeight = 'bold';
            app.ViewDatabaseBtn.BackgroundColor = [0 0 0];
            app.ViewDatabaseBtn.FontColor = [1 1 1];
            app.ViewDatabaseBtn.ButtonPushedFcn = createCallbackFcn(app, @ViewDatabaseBtnPushed, true);

            app.ScanFPBtn = uibutton(app.ControlPanel, 'push');
            app.ScanFPBtn.Position = [530 15 110 30];
            app.ScanFPBtn.Text = 'Scan Finger';
            app.ScanFPBtn.FontWeight = 'bold';
            app.ScanFPBtn.BackgroundColor = [0 0 0];
            app.ScanFPBtn.FontColor = [1 1 1];
            app.ScanFPBtn.ButtonPushedFcn = createCallbackFcn(app, @ScanFPBtnPushed, true);

            app.FusionIdentifyBtn = uibutton(app.ControlPanel, 'push');
            app.FusionIdentifyBtn.Position = [650 15 120 30];
            app.FusionIdentifyBtn.Text = 'Identify (Fusion)';
            app.FusionIdentifyBtn.FontWeight = 'bold';
            app.FusionIdentifyBtn.BackgroundColor = [0 0 0];
            app.FusionIdentifyBtn.FontColor = [1 1 1];
            app.FusionIdentifyBtn.ButtonPushedFcn = createCallbackFcn(app, @FusionIdentifyBtnPushed, true);

            app.DeleteBtn = uibutton(app.ControlPanel, 'push');
            app.DeleteBtn.Position = [780 15 120 30];
            app.DeleteBtn.Text = 'Delete Identity';
            app.DeleteBtn.FontWeight = 'bold';
            app.DeleteBtn.BackgroundColor = [0 0 0];
            app.DeleteBtn.FontColor = [1 1 1];
            app.DeleteBtn.ButtonPushedFcn = createCallbackFcn(app, @DeleteBtnPushed, true);

            % display panel
            app.DisplayPanel = uipanel(app.MainPanel);
            app.DisplayPanel.Position = [20 220 930 320];
            app.DisplayPanel.BackgroundColor = panelBg;
            app.DisplayPanel.BorderType = 'line';
            app.DisplayPanel.BorderWidth = 1;
            app.DisplayPanel.HighlightColor = [0.25 0.25 0.28];

            % listbox on the left
            app.DBList = uilistbox(app.DisplayPanel);
            app.DBList.Position = [15 20 160 280];
            app.DBList.Items = {};
            app.DBList.ValueChangedFcn = createCallbackFcn(app, @DBListValueChangedWrapper, true);
            app.DBList.BackgroundColor = [0.05 0.05 0.05];
            app.DBList.FontColor = [1 1 1];
            app.DBList.Multiselect = 'off';

            % face axes (shifted right to give space to listbox)
            app.ScanImageAxes = uiaxes(app.DisplayPanel);
            app.ScanImageAxes.Position = [190 110 330 190];
            app.ScanImageAxes.Color = darkerBg;
            title(app.ScanImageAxes, 'Scanned Photo', 'FontSize', 14, 'Color', textColor);
            app.ScanImageAxes.XTick = [];
            app.ScanImageAxes.YTick = [];
            app.ScanImageAxes.XColor = [0.3 0.3 0.35];
            app.ScanImageAxes.YColor = [0.3 0.3 0.35];
            app.ScanImageAxes.Box = 'on';

            app.FoundImageAxes = uiaxes(app.DisplayPanel);
            app.FoundImageAxes.Position = [540 110 360 190];
            app.FoundImageAxes.Color = darkerBg;
            title(app.FoundImageAxes, 'Match Result', 'FontSize', 14, 'Color', textColor);
            app.FoundImageAxes.XTick = [];
            app.FoundImageAxes.YTick = [];
            app.FoundImageAxes.XColor = [0.3 0.3 0.35];
            app.FoundImageAxes.YColor = [0.3 0.3 0.35];
            app.FoundImageAxes.Box = 'on';

            % fingerprint panel at bottom
            app.FPPanel = uipanel(app.DisplayPanel);
            app.FPPanel.Position = [190 10 710 80];
            app.FPPanel.BackgroundColor = [0.12 0.12 0.14];
            app.FPPanel.BorderType = 'line';
            app.FPPanel.Title = 'Fingerprint (linked to same identity)';
            app.FPPanel.FontSize = 11;
            app.FPPanel.ForegroundColor = [0.4 1 0.6];

            app.FPScanAxes = uiaxes(app.FPPanel);
            app.FPScanAxes.Position = [10 5 330 50];
            app.FPScanAxes.Color = [0.05 0.05 0.05];
            app.FPScanAxes.XTick = [];
            app.FPScanAxes.YTick = [];
            title(app.FPScanAxes, 'Scanned / Registered FP', 'FontSize', 10, 'Color', textColor);

            app.FPMatchAxes = uiaxes(app.FPPanel);
            app.FPMatchAxes.Position = [360 5 330 50];
            app.FPMatchAxes.Color = [0.05 0.05 0.05];
            app.FPMatchAxes.XTick = [];
            app.FPMatchAxes.YTick = [];
            title(app.FPMatchAxes, 'Matched / Stored FP', 'FontSize', 10, 'Color', textColor);

            % info panel
            app.InfoPanel = uipanel(app.MainPanel);
            app.InfoPanel.Position = [20 20 930 190];
            app.InfoPanel.BackgroundColor = panelBg;
            app.InfoPanel.BorderType = 'line';
            app.InfoPanel.BorderWidth = 1;
            app.InfoPanel.HighlightColor = [0.25 0.25 0.28];

            app.StatusLabel = uilabel(app.InfoPanel);
            app.StatusLabel.Position = [30 130 870 20];
            app.StatusLabel.Text = 'System initializing...';
            app.StatusLabel.FontSize = 13;
            app.StatusLabel.FontWeight = 'bold';
            app.StatusLabel.FontColor = textColor;

            app.MatchLabel = uilabel(app.InfoPanel);
            app.MatchLabel.Position = [30 100 600 20];
            app.MatchLabel.Text = '';
            app.MatchLabel.FontSize = 15;
            app.MatchLabel.FontWeight = 'bold';
            app.MatchLabel.FontColor = textColor;

            app.ConfidenceLabel = uilabel(app.InfoPanel);
            app.ConfidenceLabel.Position = [30 70 400 20];
            app.ConfidenceLabel.Text = '';
            app.ConfidenceLabel.FontSize = 12;
            app.ConfidenceLabel.FontColor = [0.6 0.6 0.65];

            app.DatabaseCountLabel = uilabel(app.InfoPanel);
            app.DatabaseCountLabel.Position = [650 70 250 20];
            app.DatabaseCountLabel.Text = 'Database: 0 identities';
            app.DatabaseCountLabel.FontSize = 12;
            app.DatabaseCountLabel.FontColor = accentCyan;
            app.DatabaseCountLabel.HorizontalAlignment = 'right';
        end
    end

    % =====================================================================
    % PUBLIC CTOR
    % =====================================================================
    methods (Access = public)
        function app = FacialRecognition
            createComponents(app)
            registerApp(app, app.UIFigure)
            runStartupFcn(app, @startupFcn)
        end

        function delete(app)
            delete(app.UIFigure)
        end
    end
end

% =====================================================================
% ============  FINGERPRINT HELPER FUNCTIONS  =========================
% (your code, dropped at the bottom so only ONE .m is needed)
% =====================================================================
function [E, BW, SKEL, ORIENT] = fpEnhance(I)
    if size(I,3)==3, G = rgb2gray(I); else, G = I; end
    G = im2double(G);
    G = imadjust(G);
    G = wiener2(G,[3 3]);
    orientations = 0:22.5:157.5;
    E = zeros(size(G));
    for th = orientations
        lam = 8;
        g = imgaborfilt(G, lam, th);
        E = max(E, g);
    end
    E = mat2gray(E);
    BW = imbinarize(E,'adaptive','Sensitivity',0.45);
    BW = bwareaopen(BW, 30);
    BW = imclose(BW, strel('line',3,0));
    SKEL = bwmorph(BW,'thin',Inf);
    [GX,GY] = imgradientxy(G); ORIENT = atan2(GY, GX);
end

function score = fpMatch(Iprobe, Iref)
    % allow skeleton or image
    if islogical(Iprobe) || max(Iprobe(:))==1
        S1 = Iprobe;
    else
        [~,~,S1,~] = fpEnhance(Iprobe);
    end
    if islogical(Iref) || max(Iref(:))==1
        S2 = Iref;
    else
        [~,~,S2,~] = fpEnhance(Iref);
    end
    [m1,~] = fpMinutiae(S1);
    [m2,~] = fpMinutiae(S2);
    if isempty(m1) || isempty(m2), score = 0; return; end
    maxDist = 12; maxDang = pi/6; used = false(size(m2,1),1); matches = 0;
    for i=1:size(m1,1)
        d = hypot(m2(:,1)-m1(i,1), m2(:,2)-m1(i,2));
        dang = abs(atan2(sin(m2(:,4)-m1(i,4)), cos(m2(:,4)-m1(i,4))));
        typeOk = (m2(:,3)==m1(i,3));
        cand = find(~used & typeOk & d<maxDist & dang<maxDang);
        if isempty(cand), continue; end
        [~,jrel] = min(d(cand)); j = cand(jrel);
        used(j) = true; matches = matches + 1;
    end
    normDen = max( min(size(m1,1), size(m2,1)), 1 );
    score = matches / normDen;
end

function [minu, skel] = fpMinutiae(SKEL)
    skel = SKEL>0;
    skel = bwmorph(skel,'spur',5);
    [rw,cl] = find(skel); minu = [];
    D = bwdist(~skel); [gy,gx] = gradient(D);
    for k = 1:numel(rw)
        r = rw(k); c = cl(k);
        if r<=1 || c<=1 || r>=size(skel,1)-1 || c>=size(skel,2)-1, continue; end
        nb = skel(r-1:r+1, c-1:c+1); P = nb; P(2,2)=0;
        deg = sum(P(:));
        if deg==1, type=1; elseif deg>=3, type=3; else, continue; end
        theta = atan2(gy(r,c), gx(r,c));
        minu = [minu; r, c, type, theta]; %#ok<AGROW>
    end
    if ~isempty(minu)
        keep = true(size(minu,1),1);
        for i=1:size(minu,1)
            if ~keep(i), continue; end
            d = hypot(minu(:,1)-minu(i,1), minu(:,2)-minu(i,2));
            dup = d<5; dup(i)=false; keep(dup)=false;
        end
        minu = minu(keep,:);
    end
end

//
//  ChatView.swift
//  MindVault
//
//  AI Chat interface with RAG
//

import SwiftUI
import SwiftData

struct ChatView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Conversation.updatedAt, order: .reverse) private var conversations: [Conversation]
    
    @State private var currentConversation: Conversation?
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var isLoading = false
    @State private var showingConversationList = false
    @State private var showingContext = false
    @State private var currentContext: [ChatContext] = []
    
    @StateObject private var ragService = RAGService.shared
    @StateObject private var speechRecognition = SpeechRecognitionService.shared
    @StateObject private var textToSpeech = TextToSpeechService.shared
    @StateObject private var models = ModelManager.shared
    @FocusState private var isInputFocused: Bool
    @State private var showingVoicePermissionAlert = false
    @State private var streamingText = ""
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if messages.isEmpty {
                    emptyStateView
                } else {
                    messageListView
                }

                if !models.isReady {
                    modelStatusBanner
                }

                inputBarView
            }
            .navigationTitle("AI Assistant")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showingConversationList = true
                    } label: {
                        Image(systemName: "clock.arrow.circlepath")
                    }
                }
                
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        startNewConversation()
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                }
            }
            .sheet(isPresented: $showingConversationList) {
                ConversationListView(
                    conversations: conversations,
                    onSelect: { conversation in
                        loadConversation(conversation)
                    },
                    onDelete: { conversation in
                        deleteConversation(conversation)
                    }
                )
            }
            .sheet(isPresented: $showingContext) {
                ContextView(contexts: currentContext)
                    .presentationDetents([.medium, .large])
            }
        }
    }
    
    // MARK: - Empty State
    private var emptyStateView: some View {
        ScrollView {
            VStack(spacing: 30) {
                Spacer(minLength: 40)
                
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 60))
                    .foregroundStyle(.indigo.gradient)
                
                VStack(spacing: 8) {
                    Text("Your Personal AI")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Text("I know everything you've shared in your journal. Ask me anything about your life, skills, or experiences!")
                        .font(.body)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                }
                
                // Suggested Questions
                VStack(alignment: .leading, spacing: 12) {
                    Text("Try asking:")
                        .font(.headline)
                        .padding(.horizontal)
                    
                    LazyVGrid(columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible())
                    ], spacing: 12) {
                        ForEach(SuggestedQuestion.defaults.prefix(6)) { question in
                            SuggestedQuestionCard(question: question) {
                                inputText = question.question
                                sendMessage()
                            }
                        }
                    }
                    .padding(.horizontal)
                }
                
                Spacer(minLength: 100)
            }
        }
    }
    
    // MARK: - Message List
    private var messageListView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(messages) { message in
                        MessageBubble(
                            message: message,
                            onShowContext: {
                                if let ids = message.contextIds,
                                   let snippets = message.contextSnippets,
                                   !ids.isEmpty {
                                    let titles = message.contextTitles ?? Array(repeating: "Source", count: ids.count)
                                    let types = message.contextTypes ?? Array(repeating: "document", count: ids.count)
                                    let scores = message.contextScores ?? Array(repeating: Float(0), count: ids.count)
                                    
                                    currentContext = (0..<ids.count).map { i in
                                        ChatContext(
                                            documentId: ids[i],
                                            documentType: types[i],
                                            title: titles[i],
                                            snippet: snippets[i],
                                            relevanceScore: scores[i],
                                            date: nil
                                        )
                                    }
                                    showingContext = true
                                }
                            },
                            onFeedback: { isHelpful in
                                provideFeedback(message: message, isHelpful: isHelpful)
                            }
                        )
                        .id(message.id)
                    }
                    
                    if !streamingText.isEmpty {
                        // Live, in-flight assistant response (tokens streaming in).
                        MessageBubble(
                            message: ChatMessage(
                                content: streamingText,
                                role: MessageRole.assistant.rawValue,
                                conversationId: currentConversation?.id ?? UUID()
                            ),
                            onShowContext: {},
                            onFeedback: { _ in }
                        )
                        .id("streaming")
                    } else if isLoading {
                        HStack {
                            TypingIndicator()
                            Spacer()
                        }
                        .padding(.horizontal)
                        .id("loading")
                    }
                }
                .padding()
            }
            .onChange(of: messages.count) { _, _ in
                withAnimation {
                    proxy.scrollTo(messages.last?.id, anchor: .bottom)
                }
            }
            .onChange(of: isLoading) { _, loading in
                if loading {
                    withAnimation {
                        proxy.scrollTo("loading", anchor: .bottom)
                    }
                }
            }
            .onChange(of: streamingText) { _, _ in
                withAnimation {
                    proxy.scrollTo("streaming", anchor: .bottom)
                }
            }
        }
    }
    
    // MARK: - Input Bar
    private var inputBarView: some View {
        VStack(spacing: 0) {
            Divider()
            
            HStack(alignment: .bottom, spacing: 12) {
                // Microphone button for voice input
                Button {
                    toggleVoiceInput()
                } label: {
                    Image(systemName: speechRecognition.isRecording ? "mic.fill" : "mic")
                        .font(.system(size: 24))
                        .foregroundStyle(speechRecognition.isRecording ? .red : .indigo)
                        .frame(width: 44, height: 44)
                        .background(speechRecognition.isRecording ? Color.red.opacity(0.1) : Color.indigo.opacity(0.1))
                        .clipShape(Circle())
                }
                .disabled(isLoading || !models.isReady)

                TextField("Ask me anything...", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .padding(12)
                    .background(Color(.systemGray6))
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .lineLimit(1...5)
                    .focused($isInputFocused)
                    .onChange(of: speechRecognition.recognizedText) { _, newText in
                        if speechRecognition.isRecording {
                            inputText = newText
                        }
                    }
                
                Button {
                    sendMessage()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 36))
                        .foregroundStyle(inputText.isEmpty ? .gray : .indigo)
                }
                .disabled(inputText.isEmpty || isLoading || !models.isReady)
            }
            .padding()
        }
        .background(.ultraThinMaterial)
        .alert("Microphone Access Required", isPresented: $showingVoicePermissionAlert) {
            Button("Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Please enable microphone access in Settings to use voice input.")
        }
    }
    
    // MARK: - Functions
    private func sendMessage() {
        guard !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard models.isReady else { return }

        let userMessage = inputText
        inputText = ""
        isInputFocused = false

        // Conversation history (prior turns) before we append the new message.
        let history: [(role: String, content: String)] = messages
            .suffix(10)
            .map { (role: $0.role, content: $0.content) }

        // Create or get conversation
        if currentConversation == nil {
            let conversation = Conversation(title: String(userMessage.prefix(50)))
            modelContext.insert(conversation)
            currentConversation = conversation
        }

        // Add user message
        let userChatMessage = ChatMessage(
            content: userMessage,
            role: MessageRole.user.rawValue,
            conversationId: currentConversation!.id
        )
        modelContext.insert(userChatMessage)
        messages.append(userChatMessage)

        // Stream the assistant response.
        isLoading = true
        streamingText = ""

        Task {
            var collected: [ChatContext] = []
            var accumulated = ""
            do {
                for try await event in ragService.streamResponse(to: userMessage, history: history) {
                    switch event {
                    case .sources(let contexts):
                        collected = contexts
                    case .token(let delta):
                        accumulated += delta
                        streamingText = accumulated
                        isLoading = false   // first token arrived; drop typing indicator
                    }
                }
            } catch {
                if accumulated.isEmpty {
                    accumulated = "Sorry — \(error.localizedDescription)"
                }
            }

            let assistantMessage = ChatMessage(
                content: accumulated,
                role: MessageRole.assistant.rawValue,
                conversationId: currentConversation!.id,
                contextIds: collected.map { $0.documentId },
                contextSnippets: collected.map { $0.snippet },
                contextTitles: collected.map { $0.title },
                contextTypes: collected.map { $0.documentType },
                contextScores: collected.map { $0.relevanceScore }
            )
            modelContext.insert(assistantMessage)
            messages.append(assistantMessage)

            streamingText = ""
            isLoading = false
            currentConversation?.updatedAt = Date()
            try? modelContext.save()
        }
    }

    // MARK: - Model Loading Banner
    private var modelStatusBanner: some View {
        HStack(spacing: 12) {
            if case .failed(let message) = models.phase {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Model unavailable").font(.subheadline).fontWeight(.semibold)
                    Text(message).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
                Spacer()
                Button("Retry") { Task { await models.prepare() } }
                    .buttonStyle(.bordered)
            } else {
                ProgressView()
                VStack(alignment: .leading, spacing: 2) {
                    Text(models.statusMessage.isEmpty ? "Loading on-device AI…" : models.statusMessage)
                        .font(.subheadline)
                    if models.phase == .downloading, models.downloadProgress > 0 {
                        ProgressView(value: models.downloadProgress)
                            .progressViewStyle(.linear)
                    }
                }
                Spacer()
            }
        }
        .padding(12)
        .background(.ultraThinMaterial)
    }
    
    private func startNewConversation() {
        currentConversation = nil
        messages = []
    }
    
    private func loadConversation(_ conversation: Conversation) {
        currentConversation = conversation
        
        // Fetch messages for this conversation
        let conversationId = conversation.id
        let descriptor = FetchDescriptor<ChatMessage>(
            predicate: #Predicate { $0.conversationId == conversationId },
            sortBy: [SortDescriptor(\.timestamp)]
        )
        
        do {
            messages = try modelContext.fetch(descriptor)
        } catch {
            print("Error loading messages: \(error)")
            messages = []
        }
        
        showingConversationList = false
    }
    
    private func deleteConversation(_ conversation: Conversation) {
        // Delete messages
        let conversationId = conversation.id
        let descriptor = FetchDescriptor<ChatMessage>(
            predicate: #Predicate { $0.conversationId == conversationId }
        )
        
        do {
            let messagesToDelete = try modelContext.fetch(descriptor)
            for message in messagesToDelete {
                modelContext.delete(message)
            }
        } catch {
            print("Error deleting messages: \(error)")
        }
        
        modelContext.delete(conversation)
        
        if currentConversation?.id == conversation.id {
            startNewConversation()
        }
    }
    
    private func provideFeedback(message: ChatMessage, isHelpful: Bool) {
        message.isHelpful = isHelpful
    }
    
    private func toggleVoiceInput() {
        if speechRecognition.isRecording {
            speechRecognition.stopRecording()
        } else {
            Task {
                // Check authorization
                if speechRecognition.authorizationStatus == .notDetermined {
                    let authorized = await speechRecognition.requestAuthorization()
                    if !authorized {
                        showingVoicePermissionAlert = true
                        return
                    }
                } else if speechRecognition.authorizationStatus != .authorized {
                    showingVoicePermissionAlert = true
                    return
                }
                
                // Start recording
                do {
                    try speechRecognition.startRecording()
                } catch {
                    print("Failed to start recording: \(error)")
                }
            }
        }
    }
}

// MARK: - Message Bubble
struct MessageBubble: View {
    let message: ChatMessage
    let onShowContext: () -> Void
    let onFeedback: (Bool) -> Void
    
    @StateObject private var textToSpeech = TextToSpeechService.shared
    
    var body: some View {
        HStack(alignment: .top) {
            if message.isUser {
                Spacer(minLength: 60)
            } else {
                Image(systemName: "brain.head.profile")
                    .font(.title3)
                    .foregroundStyle(.indigo)
                    .frame(width: 32, height: 32)
                    .background(.indigo.opacity(0.1))
                    .clipShape(Circle())
            }
            
            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 8) {
                Text(message.content)
                    .padding(12)
                    .background(message.isUser ? Color.indigo : Color(.systemGray6))
                    .foregroundStyle(message.isUser ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                
                if message.isAssistant {
                    HStack(spacing: 12) {
                        // Speaker button for AI responses
                        Button {
                            if textToSpeech.isSpeaking {
                                textToSpeech.stop()
                            } else {
                                textToSpeech.speak(message.content)
                            }
                        } label: {
                            Image(systemName: textToSpeech.isSpeaking ? "speaker.wave.3.fill" : "speaker.wave.2")
                                .font(.caption)
                        }
                        .foregroundStyle(textToSpeech.isSpeaking ? .indigo : .secondary)
                        
                        if message.contextSnippets?.isEmpty == false {
                            Button {
                                onShowContext()
                            } label: {
                                Label("View Sources", systemImage: "doc.text.magnifyingglass")
                                    .font(.caption)
                            }
                            .buttonStyle(.bordered)
                            .buttonBorderShape(.capsule)
                        }
                        
                        HStack(spacing: 8) {
                            Button {
                                onFeedback(true)
                            } label: {
                                Image(systemName: message.isHelpful == true ? "hand.thumbsup.fill" : "hand.thumbsup")
                                    .font(.caption)
                            }
                            .foregroundStyle(message.isHelpful == true ? .green : .secondary)
                            
                            Button {
                                onFeedback(false)
                            } label: {
                                Image(systemName: message.isHelpful == false ? "hand.thumbsdown.fill" : "hand.thumbsdown")
                                    .font(.caption)
                            }
                            .foregroundStyle(message.isHelpful == false ? .red : .secondary)
                        }
                    }
                }
            }
            
            if !message.isUser {
                Spacer(minLength: 60)
            }
        }
    }
}

// MARK: - Typing Indicator
struct TypingIndicator: View {
    @State private var animationOffset: CGFloat = 0
    
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "brain.head.profile")
                .font(.title3)
                .foregroundStyle(.indigo)
                .frame(width: 32, height: 32)
                .background(.indigo.opacity(0.1))
                .clipShape(Circle())
            
            HStack(spacing: 4) {
                ForEach(0..<3, id: \.self) { index in
                    Circle()
                        .fill(Color.gray)
                        .frame(width: 8, height: 8)
                        .offset(y: animationOffset)
                        .animation(
                            Animation.easeInOut(duration: 0.5)
                                .repeatForever()
                                .delay(Double(index) * 0.15),
                            value: animationOffset
                        )
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(.systemGray6))
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
        .onAppear {
            animationOffset = -5
        }
    }
}

// MARK: - Suggested Question Card
struct SuggestedQuestionCard: View {
    let question: SuggestedQuestion
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 8) {
                Image(systemName: question.icon)
                    .font(.title3)
                    .foregroundStyle(.indigo)
                
                Text(question.question)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .multilineTextAlignment(.leading)
                    .lineLimit(2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(Color(.systemGray6))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }
}

// MARK: - Conversation List View
struct ConversationListView: View {
    @Environment(\.dismiss) private var dismiss
    let conversations: [Conversation]
    let onSelect: (Conversation) -> Void
    let onDelete: (Conversation) -> Void
    
    var body: some View {
        NavigationStack {
            List {
                if conversations.isEmpty {
                    ContentUnavailableView(
                        "No Conversations",
                        systemImage: "bubble.left.and.bubble.right",
                        description: Text("Your chat history will appear here")
                    )
                } else {
                    ForEach(conversations) { conversation in
                        Button {
                            onSelect(conversation)
                        } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(conversation.title)
                                    .font(.headline)
                                    .foregroundStyle(.primary)
                                
                                Text(conversation.updatedAt, style: .relative)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .swipeActions(edge: .trailing) {
                            Button(role: .destructive) {
                                onDelete(conversation)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }
            }
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Context View
struct ContextView: View {
    @Environment(\.dismiss) private var dismiss
    let contexts: [ChatContext]
    
    var body: some View {
        NavigationStack {
            Group {
                if contexts.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "doc.questionmark")
                            .font(.system(size: 50))
                            .foregroundStyle(.secondary)
                        Text("No sources available")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                        Text("The AI generated this response without retrieving specific documents.")
                            .font(.subheadline)
                            .foregroundStyle(.tertiary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        ForEach(contexts, id: \.documentId) { context in
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Image(systemName: iconForType(context.documentType))
                                        .foregroundStyle(.indigo)
                                    Text(context.title.isEmpty ? "Untitled" : context.title)
                                        .font(.headline)
                                        .lineLimit(2)
                                    Spacer()
                                    Text(String(format: "%.0f%%", context.relevanceScore * 100))
                                        .font(.caption)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(Color.indigo.opacity(0.2))
                                        .cornerRadius(8)
                                }
                                
                                if !context.snippet.isEmpty {
                                    Text(context.snippet)
                                        .font(.body)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(4)
                                }
                                
                                HStack {
                                    Label(context.documentType.capitalized, systemImage: "tag")
                                        .font(.caption)
                                        .foregroundStyle(.tertiary)
                                    
                                    if let date = context.date {
                                        Spacer()
                                        Text(date, style: .date)
                                            .font(.caption)
                                            .foregroundStyle(.tertiary)
                                    }
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
            }
            .navigationTitle("Sources (\(contexts.count))")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func iconForType(_ type: String) -> String {
        switch type.lowercased() {
        case "email": return "envelope"
        case "journal": return "book"
        case "profile": return "person"
        default: return "doc.text"
        }
    }
}

#Preview {
    ChatView()
        .modelContainer(for: [ChatMessage.self, Conversation.self], inMemory: true)
}

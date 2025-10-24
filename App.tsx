import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence, useMotionValue, animate } from 'framer-motion';
import { GoogleGenAI, Modality, LiveSession, LiveServerMessage, Blob as GenAiBlob, Operation, ChatMessagePart } from '@google/genai';
import { ChatMessage, ChatSession, AudioConfig, Persona } from './types';
import getAi from './services/geminiService';
import { decodeAudioData, encode, decode } from './utils/audioUtils';
import { ICONS, PERSONAS as defaultPersonas } from './constants';


// --- Hooks ---
const useMediaQuery = (query: string) => {
    const [matches, setMatches] = useState(window.matchMedia(query).matches);
    useEffect(() => {
        const media = window.matchMedia(query);
        const listener = () => setMatches(media.matches);
        media.addEventListener('change', listener);
        return () => media.removeEventListener('change', listener);
    }, [query]);
    return matches;
};

const useLocalStorage = <T,>(key: string, initialValue: T): [T, React.Dispatch<React.SetStateAction<T>>] => {
    const [storedValue, setStoredValue] = useState<T>(() => {
        try {
            const item = window.localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        } catch (error) {
            console.error(error);
            return initialValue;
        }
    });

    const setValue = (value: T | ((val: T) => T)) => {
        try {
            const valueToStore = value instanceof Function ? value(storedValue) : value;
            setStoredValue(valueToStore);
            window.localStorage.setItem(key, JSON.stringify(valueToStore));
        } catch (error) {
            console.error(error);
        }
    };
    return [storedValue, setValue];
};


// --- Helper Types ---
type View = 'chat' | 'live' | 'media' | 'scheduler';
type LiveTranscriptEntry = { speaker: 'user' | 'model' | 'system', text: string, timestamp: string };
type ToolOutput = { type: 'search', results: { title: string, snippet: string }[] } | null;

// --- Helper Functions ---
const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
};

// --- Helper Components ---
const Loader: React.FC<{ text?: string }> = ({ text = "Thinking..." }) => (
    <div className="flex items-center space-x-2 p-2">
        <motion.div className="w-2 h-2 bg-accent-gold rounded-full" animate={{ scale: [1, 1.2, 1], transition: { duration: 0.5, repeat: Infinity } }} />
        <span className="text-text-porcelain text-sm font-medium">{text}</span>
    </div>
);

const AVAILABLE_VOICES: { id: Persona['voiceId']; name: string }[] = [
    { id: 'Kore', name: 'Kore (Female)' },
    { id: 'Puck', name: 'Puck (Male)' },
    { id: 'Zephyr', name: 'Zephyr (Female)' },
    { id: 'Charon', name: 'Charon (Male)' },
];


// --- Agent Components ---
const ChatAgent: React.FC<{
    personas: Persona[];
    setPersonas: React.Dispatch<React.SetStateAction<Persona[]>>;
}> = ({ personas, setPersonas }) => {
    const [sessions, setSessions] = useLocalStorage<ChatSession[]>('chat-sessions', []);
    const [activeSessionId, setActiveSessionId] = useLocalStorage<string | null>('active-session-id', null);
    
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [activePersonaId, setActivePersonaId] = useState<string>(personas[0].id);
    const [uploadedFile, setUploadedFile] = useState<{ file: File; preview: string } | null>(null);
    const [isListening, setIsListening] = useState(false);
    const [isPersonaModalOpen, setIsPersonaModalOpen] = useState(false);

    const [audioConfig, setAudioConfig] = useState<Omit<AudioConfig, 'voice'>>({ isPlaying: null });
    
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const recognitionRef = useRef<any | null>(null);

    const activeSession = sessions.find(s => s.id === activeSessionId);
    const activePersona = personas.find(p => p.id === activePersonaId) || personas[0];

    useEffect(() => {
        if (!activeSessionId || !sessions.some(s => s.id === activeSessionId)) {
            handleNewChat();
        }
    }, [sessions, activeSessionId]);
    
    useEffect(() => {
        chatContainerRef.current?.scrollTo({ top: chatContainerRef.current.scrollHeight, behavior: 'smooth' });
    }, [activeSession?.messages]);
    
    const updateSessionMessages = (newMessages: ChatMessage[], newTitle?: string) => {
        if (!activeSessionId) return;
        const updatedSessions = sessions.map(s => s.id === activeSessionId ? { ...s, messages: newMessages, title: newTitle || s.title } : s);
        setSessions(updatedSessions);
    };

    const handleNewChat = () => {
        const newSession: ChatSession = {
            id: `session-${Date.now()}`,
            title: "New Conversation",
            createdAt: Date.now(),
            messages: [{ role: 'model', parts: [{ text: "Welcome! How can I assist you today?" }] }]
        };
        setSessions(prev => [newSession, ...prev]);
        setActiveSessionId(newSession.id);
    };
    
    const handleSendMessage = async () => {
        if ((!input && !uploadedFile) || isLoading || !activeSession) return;
        
        const userParts: ChatMessagePart[] = [];
        if (input) userParts.push({ text: input });
        if (uploadedFile) {
            const base64Data = await blobToBase64(uploadedFile.file);
            userParts.push({ inlineData: { data: base64Data, mimeType: uploadedFile.file.type } });
        }

        const newUserMessage: ChatMessage = { role: 'user', parts: userParts };
        const updatedMessages = [...activeSession.messages, newUserMessage];
        
        let newTitle = activeSession.title;
        if(activeSession.messages.length <= 1){
            const textForTitle = input || uploadedFile?.file.name || "New Chat";
            newTitle = textForTitle.substring(0, 25) + (textForTitle.length > 25 ? "..." : "");
        }
        updateSessionMessages(updatedMessages, newTitle);

        setInput('');
        setUploadedFile(null);
        setIsLoading(true);

        try {
            const ai = getAi();
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-pro',
                contents: updatedMessages.map(m => ({ role: m.role, parts: m.parts })),
                config: { systemInstruction: activePersona.prompt }
            });

            const modelMessage: ChatMessage = { role: 'model', parts: [{ text: response.text }] };
            updateSessionMessages([...updatedMessages, modelMessage], newTitle);

        } catch (error) {
            console.error(error);
            const errorMessage: ChatMessage = { role: 'model', parts: [{ text: "Sorry, I encountered an error. Please try again." }] };
            updateSessionMessages([...updatedMessages, errorMessage], newTitle);
        } finally {
            setIsLoading(false);
        }
    };
    
    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const preview = URL.createObjectURL(file);
            setUploadedFile({ file, preview });
        }
    };

    const handleToggleListening = () => {
        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert("Sorry, your browser doesn't support speech recognition.");
            return;
        }

        if (isListening) {
            recognitionRef.current?.stop();
            return;
        }

        const recognition = new SpeechRecognition();
        recognitionRef.current = recognition;
        recognition.interimResults = true;
        recognition.continuous = true;

        recognition.onstart = () => setIsListening(true);
        recognition.onend = () => setIsListening(false);
        recognition.onerror = (event: any) => {
            console.error('Speech recognition error:', event.error);
            setIsListening(false);
        };
        
        recognition.onresult = (event: any) => {
            const transcript = Array.from(event.results)
                .map((result: any) => result[0])
                .map((result) => result.transcript)
                .join('');
            setInput(transcript);
        };
        
        recognition.start();
    };


    const handleSpeak = (text: string, index: number) => {
        generateAndPlayAudio(text, index, activePersona.voiceId);
    };
        
    const generateAndPlayAudio = async (text: string, index: number, voice: string) => {
        setAudioConfig(prev => ({ ...prev, isPlaying: index.toString() }));
        try {
            const ai = getAi();
            const response = await ai.models.generateContent({
                model: "gemini-2.5-flash-preview-tts",
                contents: [{ parts: [{ text }] }],
                config: {
                    responseModalities: [Modality.AUDIO],
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } },
                },
            });

            const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            if (!base64Audio) throw new Error("No audio data returned.");

            if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
                audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
            }
            const audioBuffer = await decodeAudioData(decode(base64Audio), audioContextRef.current, 24000, 1);
            const source = audioContextRef.current.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContextRef.current.destination);
            source.start();
            source.onended = () => setAudioConfig(prev => ({...prev, isPlaying: null }));

        } catch (error) {
            console.error("TTS Error:", error);
            setAudioConfig(prev => ({...prev, isPlaying: null }));
        }
    };
    
    const renderMessageContent = (msg: ChatMessage, msgIndex: number) => (
         <div className="chat-bubble-content flex flex-col items-start gap-2">
            {msg.parts.map((part, partIndex) => {
                if (part.text) {
                    return <p key={`${msgIndex}-${partIndex}`} className="whitespace-pre-wrap">{part.text}</p>;
                }
                if (part.inlineData) {
                    const src = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                    if (part.inlineData.mimeType.startsWith('image/')) {
                        return <img key={`${msgIndex}-${partIndex}`} src={src} alt="user upload" className="max-w-xs rounded-lg border border-border-subtle" />;
                    }
                    if (part.inlineData.mimeType.startsWith('video/')) {
                        return <video key={`${msgIndex}-${partIndex}`} src={src} controls className="max-w-xs rounded-lg" />;
                    }
                    if (part.inlineData.mimeType.startsWith('audio/')) {
                        return <audio key={`${msgIndex}-${partIndex}`} src={src} controls className="w-full" />;
                    }
                }
                return null;
            })}
            {msg.role === 'model' && msg.parts.some(p => p.text) && (
                <div className="mt-1 -ml-1">
                    <button onClick={() => handleSpeak(msg.parts.filter(p => p.text).map(p => p.text).join(' '), msgIndex)} disabled={audioConfig.isPlaying === msgIndex.toString()} className="p-1 rounded-full text-text-porcelain hover:text-accent-gold disabled:opacity-50 transition-colors">
                        {ICONS.SPEAKER_WAVE}
                    </button>
                </div>
            )}
        </div>
    );
    
    return (
        <div className="flex flex-col h-full bg-bg-charcoal/50">
            <PersonaModal 
                isOpen={isPersonaModalOpen}
                onClose={() => setIsPersonaModalOpen(false)}
                personas={personas}
                setPersonas={setPersonas}
                activePersonaId={activePersonaId}
                onSelectPersona={(id) => {
                    setActivePersonaId(id);
                    setIsPersonaModalOpen(false);
                }}
            />
            <input type="file" ref={fileInputRef} onChange={handleFileSelect} className="hidden" />
            <div className="p-4 flex-shrink-0 flex items-center space-x-3 border-b border-border-subtle">
                {ICONS.CHAT}
                <h2 className="text-lg font-semibold truncate">{activeSession?.title || "Chat"}</h2>
            </div>
            <div ref={chatContainerRef} className="flex-1 p-4 md:p-6 space-y-4 overflow-y-auto">
                {activeSession?.messages.map((msg, index) => (
                    <motion.div key={index} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
                        <div className={`flex items-start gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            {msg.role === 'model' && <div className="w-8 h-8 rounded-full bg-bg-charcoal/80 flex items-center justify-center text-accent-gold flex-shrink-0 mt-1">{ICONS.LOGO}</div>}
                            <div className={`max-w-md lg:max-w-xl p-3 px-4 rounded-xl ${msg.role === 'user' ? 'bg-accent-gold text-bg-onyx' : 'bg-bg-charcoal'}`}>
                                {renderMessageContent(msg, index)}
                            </div>
                        </div>
                    </motion.div>
                ))}
                {isLoading && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}><Loader /></motion.div>}
            </div>
            <footer className="p-4 flex-shrink-0 border-t border-border-subtle">
                <div className="w-full max-w-2xl mx-auto space-y-3">
                    <div className="flex justify-center">
                        <button 
                            onClick={() => setIsPersonaModalOpen(true)}
                            className="px-4 py-1.5 text-sm bg-bg-charcoal border border-border-subtle rounded-full text-text-porcelain hover:border-accent-gold hover:text-accent-gold transition-colors flex items-center gap-2"
                        >
                           {ICONS.USER}
                           {activePersona.name}
                        </button>
                    </div>

                    {uploadedFile && (
                        <div className="relative w-fit bg-bg-charcoal p-2 rounded-lg">
                            {uploadedFile.file.type.startsWith('image/') ? (
                                <img src={uploadedFile.preview} alt="preview" className="h-20 w-auto rounded" />
                            ) : (
                                <div className="h-20 flex items-center gap-2 px-2">
                                    {ICONS.DOCUMENT}
                                    <span className="text-sm text-text-porcelain truncate max-w-xs">{uploadedFile.file.name}</span>
                                </div>
                            )}
                            <button onClick={() => setUploadedFile(null)} className="absolute -top-2 -right-2 bg-bg-onyx rounded-full p-0.5 text-text-porcelain hover:text-white">
                                {ICONS.X_MARK}
                            </button>
                        </div>
                    )}

                    <form onSubmit={(e) => { e.preventDefault(); handleSendMessage(); }}>
                        <div className="relative flex items-center glass-surface rounded-xl p-2">
                            <button type="button" onClick={() => fileInputRef.current?.click()} className="p-2 text-text-porcelain hover:text-accent-gold transition-colors">
                                {ICONS.PAPERCLIP}
                            </button>
                            <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask anything..." onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); }}} className="flex-1 bg-transparent px-2 text-text-platinum focus:outline-none resize-none h-12 max-h-40" disabled={isLoading} />
                            <button type="button" onClick={handleToggleListening} className={`p-2 transition-colors ${isListening ? 'text-red-500 animate-pulse' : 'text-text-porcelain hover:text-accent-gold'}`}>
                                {ICONS.MIC}
                            </button>
                            <motion.button type="submit" disabled={isLoading || (!input && !uploadedFile)} className="p-2 rounded-full bg-accent-gold text-bg-onyx disabled:opacity-50" whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>{ICONS.SEND}</motion.button>
                        </div>
                    </form>
                </div>
            </footer>
        </div>
    );
};

const LiveAvatar: React.FC<{ state: 'idle' | 'listening' | 'speaking' }> = ({ state }) => {
    return (
        <div className="live-avatar" data-state={state}>
            <svg viewBox="0 0 80 80" className="avatar-svg">
                <defs>
                    <radialGradient id="faceGradient" cx="0.5" cy="0.4" r="0.6">
                        <stop offset="0%" stopColor="#FAD972" />
                        <stop offset="100%" stopColor="#F5C542" />
                    </radialGradient>
                    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                        <feDropShadow dx="0" dy="2" stdDeviation="2" floodColor="#000000" floodOpacity="0.3"/>
                    </filter>
                </defs>
                <circle cx="40" cy="40" r="38" fill="url(#faceGradient)" filter="url(#shadow)" className="avatar-face-body" />
                
                <g className="avatar-eyes">
                    <path d="M 28 38 C 30 32, 36 32, 38 38" stroke="#333" strokeWidth="3" fill="none" strokeLinecap="round" />
                    <path d="M 52 38 C 54 32, 60 32, 62 38" stroke="#333" strokeWidth="3" fill="none" strokeLinecap="round" />
                </g>
                
                <g className="avatar-mouth">
                    <path className="mouth-idle" d="M 30 55 Q 40 65 50 55" stroke="#333" strokeWidth="3" fill="none" strokeLinecap="round" />
                    <path className="mouth-listening" d="M 35 60 L 45 60" stroke="#333" strokeWidth="3" fill="none" strokeLinecap="round" />
                    <g className="mouth-speaking">
                        <path d="M 32 58 C 36 52, 44 52, 48 58" stroke="#333" strokeWidth="3" fill="none" strokeLinecap="round" />
                        <path d="M 32 58 C 36 65, 44 65, 48 58" stroke="#333" strokeWidth="3" fill="none" strokeLinecap="round" />
                        <path d="M 32 58 C 36 70, 44 70, 48 58" stroke="#333" strokeWidth="3" fill="none" strokeLinecap="round" />
                    </g>
                </g>
            </svg>
        </div>
    );
};


const ToolOutputCard: React.FC<{ output: ToolOutput, onClose: () => void }> = ({ output, onClose }) => (
    <AnimatePresence>
        {output && output.type === 'search' && (
            <motion.div 
                className="tool-output-card glass-surface rounded-xl p-4 flex flex-col"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
            >
                <div className="flex justify-between items-center mb-2">
                    <h3 className="font-bold text-lg flex items-center gap-2">{ICONS.SEARCH} Web Results</h3>
                    <button onClick={onClose} className="p-1 rounded-full hover:bg-white/10">{ICONS.X_MARK}</button>
                </div>
                <div className="flex-1 overflow-y-auto space-y-3 pr-2">
                    {output.results.map((res, i) => (
                        <div key={i} className="border-b border-border-subtle pb-2">
                            <h4 className="font-semibold text-text-platinum">{res.title}</h4>
                            <p className="text-sm text-text-porcelain">{res.snippet}</p>
                        </div>
                    ))}
                </div>
            </motion.div>
        )}
    </AnimatePresence>
);

const LiveAgent: React.FC<{
    personas: Persona[];
    setPersonas: React.Dispatch<React.SetStateAction<Persona[]>>;
}> = ({ personas, setPersonas }) => {
    const [isSessionActive, setIsSessionActive] = useState(false);
    const [conversationState, setConversationState] = useState<'idle' | 'listening' | 'speaking'>('idle');
    const [transcript, setTranscript] = useState<LiveTranscriptEntry[]>([]);
    const [activePersonaId, setActivePersonaId] = useState<string>(personas[0].id);
    const [isPersonaModalOpen, setIsPersonaModalOpen] = useState(false);
    const [isToolsModalOpen, setIsToolsModalOpen] = useState(false);
    const [toolOutput, setToolOutput] = useState<ToolOutput>(null);

    const sessionPromiseRef = useRef<Promise<LiveSession> | null>(null);
    const inputAudioContextRef = useRef<AudioContext | null>(null);
    const outputAudioContextRef = useRef<AudioContext | null>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const nextStartTimeRef = useRef<number>(0);
    const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
    const isNewTurnRef = useRef(true);

    const constraintsRef = useRef<HTMLDivElement>(null);
    const lastPositionRef = useRef({ x: 0, y: 0 });
    const avatarX = useMotionValue(0);
    const avatarY = useMotionValue(0);
    
    const activePersona = personas.find(p => p.id === activePersonaId) || personas[0];

    const startConversation = useCallback(async () => {
        setIsSessionActive(true);
        setConversationState('listening');
        setTranscript([]);
        nextStartTimeRef.current = 0;
        isNewTurnRef.current = true;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaStreamRef.current = stream;

            inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
            outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
            
            const ai = getAi();
            sessionPromiseRef.current = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                config: {
                    responseModalities: [Modality.AUDIO],
                    inputAudioTranscription: {},
                    outputAudioTranscription: {},
                    systemInstruction: activePersona.prompt,
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: activePersona.voiceId } } },
                },
                callbacks: {
                    onopen: () => {
                        const source = inputAudioContextRef.current!.createMediaStreamSource(stream);
                        const scriptProcessor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
                        scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
                            const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                            const pcmBlob: GenAiBlob = {
                                data: encode(new Uint8Array(new Int16Array(inputData.map(f => f * 32768)).buffer)),
                                mimeType: 'audio/pcm;rate=16000',
                            };
                            sessionPromiseRef.current?.then(session => session.sendRealtimeInput({ media: pcmBlob }));
                        };
                        source.connect(scriptProcessor);
                        scriptProcessor.connect(inputAudioContextRef.current!.destination);
                    },
                    onmessage: async (message: LiveServerMessage) => {
                        const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                        if (base64Audio) {
                            setConversationState('speaking');
                            const outCtx = outputAudioContextRef.current;
                            if (!outCtx || outCtx.state === 'closed') return;

                            nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outCtx.currentTime);
                            const audioBuffer = await decodeAudioData(decode(base64Audio), outCtx, 24000, 1);
                            const source = outCtx.createBufferSource();
                            source.buffer = audioBuffer;
                            source.connect(outCtx.destination);
                            source.start(nextStartTimeRef.current);
                            nextStartTimeRef.current += audioBuffer.duration;
                            sourcesRef.current.add(source);
                            source.onended = () => {
                                sourcesRef.current.delete(source);
                                if(sourcesRef.current.size === 0) setConversationState('listening');
                            };
                        }

                        if (message.serverContent?.outputTranscription) {
                            const newText = message.serverContent.outputTranscription.text;
                             const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                            setTranscript(prev => {
                                if (isNewTurnRef.current) {
                                    isNewTurnRef.current = false;
                                    return [...prev, { speaker: 'model', text: newText, timestamp }];
                                }
                                const lastEntry = prev[prev.length - 1];
                                if (lastEntry?.speaker === 'model') {
                                    lastEntry.text += newText;
                                    return [...prev]; // Return new array to trigger re-render
                                }
                                return [...prev, { speaker: 'model', text: newText, timestamp }];
                            });
                        }

                         if (message.serverContent?.interrupted) {
                             sourcesRef.current.forEach(source => source.stop());
                             sourcesRef.current.clear();
                             nextStartTimeRef.current = 0;
                             setConversationState('listening');
                         }
                        
                         if (message.serverContent?.turnComplete) {
                             isNewTurnRef.current = true;
                         }

                    },
                    onclose: () => console.log('Session closed.'),
                    onerror: (e) => console.error('Session error:', e),
                }
            });

        } catch (err) {
            console.error('Error starting conversation:', err);
            stopConversation();
        }
    }, [activePersona]);

    const stopConversation = useCallback(() => {
        sessionPromiseRef.current?.then(session => session.close());
        sessionPromiseRef.current = null;
        
        mediaStreamRef.current?.getTracks().forEach(track => track.stop());
        mediaStreamRef.current = null;
        
        sourcesRef.current.forEach(source => source.stop());
        sourcesRef.current.clear();

        if (inputAudioContextRef.current && inputAudioContextRef.current.state !== 'closed') {
            inputAudioContextRef.current.close();
        }
        if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') {
            outputAudioContextRef.current.close();
        }

        setIsSessionActive(false);
        setConversationState('idle');
    }, []);

    useEffect(() => {
        return () => {
            if (isSessionActive) stopConversation();
        };
    }, [isSessionActive, stopConversation]);

    const handleSelectPersona = (id: string) => {
        setActivePersonaId(id);
        setIsPersonaModalOpen(false);
        if (isSessionActive) {
            stopConversation();
            setTimeout(() => startConversation(), 100);
        }
    };
    
    const handleToolSelect = (tool: 'search' | 'image') => {
        setIsToolsModalOpen(false);
        if (!isSessionActive) return;

        if (tool === 'search') {
             const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            setTranscript(prev => [...prev, { speaker: 'system', text: `Using Web Search tool...`, timestamp }]);
            
            if (constraintsRef.current) {
                lastPositionRef.current = { x: avatarX.get(), y: avatarY.get() };
                const targetX = constraintsRef.current.offsetWidth - 120;
                animate(avatarX, targetX, { type: 'spring', stiffness: 200, damping: 20 });
                animate(avatarY, 20, { type: 'spring', stiffness: 200, damping: 20 });
            }
            
            setTimeout(() => {
                setToolOutput({
                    type: 'search',
                    results: [
                        { title: "Gemini API - Google AI for Developers", snippet: "Harness the power of Google's largest and most capable AI model. The Gemini API gives you access to the latest models from Google." },
                        { title: "What is Gemini? Everything you need to know", snippet: "Gemini is a family of multimodal large language models developed by Google DeepMind, serving as the successor to LaMDA and PaLM 2." }
                    ]
                });
            }, 1000);
        }
    };

    const closeToolOutput = () => {
        setToolOutput(null);
        animate(avatarX, lastPositionRef.current.x, { type: 'spring', stiffness: 200, damping: 20 });
        animate(avatarY, lastPositionRef.current.y, { type: 'spring', stiffness: 200, damping: 20 });
    };
    
    const ControlButton: React.FC<{ icon: React.ReactNode, label: string, onClick?: () => void, isDisabled?: boolean }> = ({ icon, label, onClick, isDisabled }) => (
        <button onClick={onClick} disabled={isDisabled} className="flex flex-col items-center justify-center space-y-1 text-text-porcelain hover:text-accent-gold transition-colors disabled:opacity-50 disabled:hover:text-text-porcelain interactive-glow p-2 rounded-lg">
            {icon}
            <span className="text-xs font-medium">{label}</span>
        </button>
    );

    return (
        <div ref={constraintsRef} className="flex flex-col h-full items-center justify-between text-center relative overflow-hidden">
            <PersonaModal 
                isOpen={isPersonaModalOpen}
                onClose={() => setIsPersonaModalOpen(false)}
                personas={personas}
                setPersonas={setPersonas}
                activePersonaId={activePersonaId}
                onSelectPersona={handleSelectPersona}
            />
            <ToolOutputCard output={toolOutput} onClose={closeToolOutput} />

            <AnimatePresence>
            {isSessionActive && (
                 <motion.div
                    drag
                    dragConstraints={constraintsRef}
                    className="absolute top-0 left-0"
                    style={{ x: avatarX, y: avatarY }}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                 >
                    <LiveAvatar state={conversationState} />
                 </motion.div>
            )}
            </AnimatePresence>

            <div className="w-full max-w-2xl flex-1 bg-bg-charcoal/50 rounded-lg p-4 my-4 overflow-y-auto text-left space-y-3">
                {transcript.filter(line => line.text.trim()).map((line, index) => (
                    <motion.div key={index} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.2 }} className="flex items-start gap-3">
                         <span className="text-xs text-text-porcelain/60 font-mono pt-1 flex-shrink-0">{line.timestamp}</span>
                         <p className={`whitespace-pre-wrap ${line.speaker === 'system' ? 'text-accent-gold italic' : 'text-text-platinum font-medium'}`}>
                            {line.text}
                        </p>
                    </motion.div>
                ))}
                {!isSessionActive && transcript.length === 0 && <p className="text-text-porcelain text-center pt-16">Start a conversation to see the transcript.</p>}
            </div>
           
             <div className="w-full max-w-md p-2 mb-4 glass-surface rounded-full flex items-center justify-around">
                <ControlButton icon={ICONS.USER} label="Persona" onClick={() => setIsPersonaModalOpen(true)} isDisabled={isSessionActive} />
                <ControlButton icon={ICONS.SCREEN_SHARE} label="Share" isDisabled />
                
                <motion.button 
                    onClick={isSessionActive ? stopConversation : startConversation}
                    className={`w-20 h-20 rounded-full flex items-center justify-center text-text-platinum transition-colors duration-300 interactive-glow ${isSessionActive ? 'bg-red-500/50' : 'bg-accent-gold/50'}`}
                    whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                >
                    <span className="text-lg font-bold">{isSessionActive ? 'Stop' : 'Start'}</span>
                </motion.button>

                <ControlButton icon={ICONS.VIDEO_ON} label="Video" isDisabled />
                <ControlButton icon={ICONS.LIVE_TOOLS} label="Tools" onClick={() => setIsToolsModalOpen(true)} isDisabled={!isSessionActive}/>
            </div>

            <AnimatePresence>
                {isToolsModalOpen && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 bg-black/60 flex items-center justify-center z-50">
                    <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} exit={{ scale: 0.9 }} className="w-full max-w-sm glass-surface rounded-xl p-6">
                       <div className="flex items-center justify-between mb-6">
                           <h3 className="text-lg font-bold">Live Tools</h3>
                           <button onClick={() => setIsToolsModalOpen(false)}>{ICONS.X_MARK}</button>
                       </div>
                       <div className="grid grid-cols-2 gap-4">
                           <button onClick={() => handleToolSelect('search')} className="flex flex-col items-center p-4 space-y-2 bg-bg-charcoal/50 rounded-lg hover:bg-accent-gold/20 transition-colors">
                               {ICONS.SEARCH}
                               <span>Web Search</span>
                           </button>
                           <button onClick={() => handleToolSelect('image')} className="flex flex-col items-center p-4 space-y-2 bg-bg-charcoal/50 rounded-lg hover:bg-accent-gold/20 transition-colors">
                               {ICONS.IMAGE_GEN}
                               <span>Image Gen</span>
                           </button>
                       </div>
                    </motion.div>
                </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

// --- Media Suite ---

const ToolkitAccordion: React.FC<{ title: string; children: React.ReactNode; defaultOpen?: boolean }> = ({ title, children, defaultOpen = false }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div className="border-b border-border-subtle">
            <button onClick={() => setIsOpen(!isOpen)} className="w-full flex justify-between items-center text-left py-3 px-1">
                <span className="font-semibold text-text-platinum">{title}</span>
                 <motion.div animate={{ rotate: isOpen ? 90 : 0 }} className="text-text-porcelain">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4"><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" /></svg>
                </motion.div>
            </button>
            <AnimatePresence>
                {isOpen && (
                    <motion.div 
                        initial="collapsed" animate="open" exit="collapsed"
                        variants={{ open: { opacity: 1, height: 'auto' }, collapsed: { opacity: 0, height: 0 } }}
                        transition={{ duration: 0.3, ease: 'easeInOut' }}
                        className="overflow-hidden"
                    >
                        <div className="pb-4 px-1">{children}</div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};


const ImageStudio: React.FC = () => {
    const [prompt, setPrompt] = useState('');
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [aspectRatio, setAspectRatio] = useState('1:1');
    const [activePreset, setActivePreset] = useState<string | null>(null);
    const [animatingPreset, setAnimatingPreset] = useState<string | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const promptInputRef = useRef<HTMLTextAreaElement>(null);
    const mode = uploadedImage ? 'edit' : 'generate';

    useEffect(() => {
        const textarea = promptInputRef.current;
        if (textarea) {
            textarea.style.height = 'auto'; // Reset height
            const lineHeight = parseFloat(getComputedStyle(textarea).lineHeight);
            const maxHeight = lineHeight * 4;
            
            if (textarea.scrollHeight > maxHeight) {
                textarea.style.height = `${maxHeight}px`;
                textarea.style.overflowY = 'auto';
            } else {
                textarea.style.height = `${textarea.scrollHeight}px`;
                textarea.style.overflowY = 'hidden';
            }
        }
    }, [prompt]);

    const PRESET_WORKFLOWS = {
        'Quick Enhance': [
            { name: 'Auto-Polish', prompt: 'Subtly enhance the overall quality of this image. Improve lighting, color balance, and sharpness without making it look unnatural. Clean up minor blemishes.' },
            { name: 'Portrait Glow', prompt: 'Give this portrait a professional, flattering look. Soften the skin slightly while retaining texture, brighten the eyes, and add a subtle, warm glow. Do not change facial features.' },
            { name: 'Vibrant Colors', prompt: 'Boost the color saturation and vibrancy of this image to make it pop, while keeping the tones realistic and natural-looking.' },
        ],
        'Professional Profiles': [
            { name: 'LinkedIn Headshot', prompt: 'Transform this into a professional headshot suitable for LinkedIn. Replace the background with a clean, modern, slightly out-of-focus office or neutral studio setting. Adjust lighting to be professional and approachable.' },
            { name: 'Corporate Look', prompt: 'Adjust the lighting and color grading to give this image a sharp, clean, and professional corporate feel.' },
        ],
        'Social & Dating': [
            { name: 'Dating Profile Boost', prompt: 'Enhance this photo for a dating profile. Improve lighting to be more flattering, add a subtle warm filter, and slightly blur the background to make the subject stand out. Keep it looking natural, authentic, and attractive.' },
            { name: 'Golden Hour Vibe', prompt: 'Apply a beautiful, warm, and dreamy "golden hour" lighting effect to this image, as if it were taken shortly before sunset.' },
        ],
        'Restore & Fix': [
            { name: 'Restore Old Photo', prompt: 'Restore this old photograph. Please improve clarity, reduce noise, fix minor scratches or dust, and enhance the colors if it\'s a color photo. If black and white, improve the contrast and tonal range.' },
            { name: 'Fix Blurriness', prompt: 'Attempt to deblur and sharpen this image, making the main subject significantly clearer and more in-focus.' },
        ],
        'Creative & Fun': [
            { name: 'Change Background', prompt: 'Change the background to: ' },
            { name: 'Retro Film Look', prompt: 'Apply a retro film look to this image, with grainy texture, slightly faded colors, and a vintage color palette.' },
        ]
    };
    
    const ASPECT_RATIOS = [
        { ratio: "1:1", icon: ICONS.ASPECT_1_1 },
        { ratio: "16:9", icon: ICONS.ASPECT_16_9 },
        { ratio: "9:16", icon: ICONS.ASPECT_9_16 },
        { ratio: "4:3", icon: ICONS.ASPECT_4_3 },
        { ratio: "3:4", icon: ICONS.ASPECT_3_4 },
    ];

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setUploadedImage(reader.result as string);
                setGeneratedImage(null); // Clear previous generation when new image is uploaded
                setActivePreset(null);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleGenerate = async () => {
        if (!prompt) {
            setError("Please enter a prompt.");
            return;
        }
        setIsLoading(true);
        setGeneratedImage(null);
        setError(null);
        setActivePreset(null);

        try {
            const ai = getAi();
            if (mode === 'edit' && uploadedImage) {
                const base64Data = uploadedImage.split(',')[1];
                const mimeType = uploadedImage.match(/data:(.*);/)?.[1] || 'image/png';
                const response = await ai.models.generateContent({
                    model: 'gemini-2.5-flash-image',
                    contents: { parts: [{ inlineData: { data: base64Data, mimeType } }, { text: prompt }] },
                    config: { responseModalities: [Modality.IMAGE] }
                });
                for (const part of response.candidates[0].content.parts) {
                    if (part.inlineData) {
                        setGeneratedImage(`data:${part.inlineData.mimeType};base64,${part.inlineData.data}`);
                    }
                }
            } else {
                const response = await ai.models.generateImages({
                    model: 'imagen-4.0-generate-001',
                    prompt: prompt,
                    config: {
                        numberOfImages: 1,
                        aspectRatio: aspectRatio as "1:1" | "3:4" | "4:3" | "9:16" | "16:9",
                        outputMimeType: 'image/png'
                    }
                });
                const base64ImageBytes = response.generatedImages[0].image.imageBytes;
                setGeneratedImage(`data:image/png;base64,${base64ImageBytes}`);
            }
        } catch (err) {
            console.error("Image generation error:", err);
            setError("An error occurred during image generation. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };
    
    const handleClear = () => {
        setUploadedImage(null);
        setGeneratedImage(null);
        setPrompt('');
        setError(null);
        setActivePreset(null);
        if(fileInputRef.current) fileInputRef.current.value = '';
    }

    const handlePresetClick = (presetName: string, presetPrompt: string) => {
        setPrompt(presetPrompt);
        setActivePreset(presetName);
        setAnimatingPreset(presetName);
        setTimeout(() => setAnimatingPreset(null), 400); // Animation duration
        setTimeout(() => {
            promptInputRef.current?.focus();
            if (presetPrompt.endsWith(': ')) {
                promptInputRef.current?.setSelectionRange(presetPrompt.length, presetPrompt.length);
            }
        }, 50);
    }

    const ImageDisplay = () => (
        <div className="w-full h-full bg-bg-onyx rounded-lg flex items-center justify-center overflow-hidden relative border border-border-subtle">
            {isLoading && <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10"><Loader text="Creating..." /></div>}
            {generatedImage ? <img src={generatedImage} alt="Generated" className="w-full h-full object-contain" /> :
             uploadedImage ? <img src={uploadedImage} alt="Uploaded" className="w-full h-full object-contain" /> :
             <div className="text-center text-text-porcelain p-4">
                 <p className="font-semibold mb-2">Welcome to the Image Studio</p>
                 <p className="text-sm">Upload an image to start editing with smart workflows, or write a prompt to generate a new image from scratch.</p>
             </div>
            }
        </div>
    );
    
    return (
        <div className="h-full flex flex-col md:flex-row">
             <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" className="hidden" />
            <aside className="w-full md:w-80 xl:w-96 p-4 border-b md:border-b-0 md:border-r border-border-subtle flex-shrink-0 overflow-y-auto">
                <h3 className="font-bold text-lg mb-4">Image Toolkit</h3>
                
                {mode === 'generate' && (
                    <ToolkitAccordion title="Generation Settings" defaultOpen>
                         <label className="text-sm font-semibold text-text-porcelain">Aspect Ratio</label>
                        <div className="flex items-center justify-between gap-2 mt-2">
                           {ASPECT_RATIOS.map(({ratio, icon}) => (
                                <button key={ratio} onClick={() => setAspectRatio(ratio)} title={ratio} className={`p-2 flex-1 h-10 flex items-center justify-center rounded-md transition-colors ${aspectRatio === ratio ? 'bg-accent-gold text-bg-onyx' : 'bg-bg-onyx hover:bg-bg-charcoal'}`}>
                                    {icon}
                                </button>
                           ))}
                        </div>
                    </ToolkitAccordion>
                )}
                
                {Object.entries(PRESET_WORKFLOWS).map(([category, presets]) => (
                    <ToolkitAccordion key={category} title={category} defaultOpen={category === 'Quick Enhance'}>
                         <div className="grid grid-cols-2 gap-2">
                            {presets.map(preset => (
                                <button 
                                    key={preset.name} 
                                    onClick={() => handlePresetClick(preset.name, preset.prompt)} 
                                    disabled={mode === 'generate'} 
                                    className={`p-3 text-sm text-left rounded-md transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-bg-onyx
                                        ${activePreset === preset.name ? 'bg-accent-gold/20 ring-1 ring-accent-gold text-accent-gold' : 'bg-bg-onyx hover:bg-bg-charcoal'}
                                        ${animatingPreset === preset.name ? 'animate-flash' : ''}
                                    `}>
                                    {preset.name}
                                </button>
                            ))}
                        </div>
                    </ToolkitAccordion>
                ))}

            </aside>
            <main className="flex-1 p-4 flex flex-col">
                <div className="flex-1 min-h-0">
                    <ImageDisplay />
                </div>
                 <div className="flex-shrink-0 pt-4 space-y-2">
                     <div className="flex items-center space-x-2 mb-2">
                         <button onClick={() => fileInputRef.current?.click()} className="px-4 py-2 bg-bg-charcoal border border-border-subtle rounded-lg text-sm interactive-glow">
                             {uploadedImage ? 'Change Image' : 'Upload Image'}
                         </button>
                         {uploadedImage && <button onClick={handleClear} className="px-4 py-2 bg-bg-charcoal border border-border-subtle rounded-lg text-sm interactive-glow">Clear</button>}
                         {generatedImage && <a href={generatedImage} download="generated-image.png" className="px-4 py-2 bg-bg-charcoal border border-border-subtle rounded-lg text-sm interactive-glow">Download</a>}
                     </div>
                     <div className="flex items-center space-x-2 glass-surface rounded-lg p-2">
                        <textarea ref={promptInputRef} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={mode === 'edit' ? "Select a workflow or describe your edits..." : "Describe the image you want to create..."} className="flex-1 bg-transparent p-2 focus:outline-none resize-none overflow-hidden" rows={1}/>
                        <button onClick={handleGenerate} disabled={isLoading || (mode === 'edit' && !uploadedImage)} className="px-6 py-2 bg-accent-gold text-bg-onyx rounded-lg font-semibold interactive-glow disabled:opacity-50">{isLoading ? "..." : "Generate"}</button>
                     </div>
                     {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
                 </div>
            </main>
        </div>
    );
};

const VideoStudio: React.FC = () => {
    const [apiKeySelected, setApiKeySelected] = useState(false);
    const [isCheckingApiKey, setIsCheckingApiKey] = useState(true);
    const [prompt, setPrompt] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');
    const [generatedVideoUrl, setGeneratedVideoUrl] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [aspectRatio, setAspectRatio] = useState<'16:9' | '9:16'>('16:9');
    const [resolution, setResolution] = useState<'720p' | '1080p'>('720p');
    
    const LOADING_MESSAGES = ['Warming up the digital director...', 'Assembling the storyboards...', 'Rendering pixels into motion...', 'This can take a few minutes...', 'Finalizing the cinematic cut...'];

    useEffect(() => {
        const checkKey = async () => {
            setIsCheckingApiKey(true);
            const hasKey = await (window as any).aistudio?.hasSelectedApiKey();
            setApiKeySelected(hasKey);
            setIsCheckingApiKey(false);
        };
        checkKey();
    }, []);

    useEffect(() => {
        let interval: number;
        if (isLoading) {
            setLoadingMessage(LOADING_MESSAGES[0]);
            let i = 1;
            interval = window.setInterval(() => {
                setLoadingMessage(LOADING_MESSAGES[i % LOADING_MESSAGES.length]);
                i++;
            }, 3000);
        }
        return () => clearInterval(interval);
    }, [isLoading]);

    const handleSelectKey = async () => {
        await (window as any).aistudio?.openSelectKey();
        setApiKeySelected(true); // Optimistic update
    };

    const handleGenerate = async () => {
        if (!prompt) {
            setError("Please enter a prompt to generate the video.");
            return;
        }
        setIsLoading(true);
        setError(null);
        setGeneratedVideoUrl(null);
        
        try {
            const ai = getAi(); // Get new instance with latest key
            let operation = await ai.models.generateVideos({
                model: 'veo-3.1-fast-generate-preview',
                prompt: prompt,
                config: { numberOfVideos: 1, resolution, aspectRatio }
            });

            while (!operation.done) {
                await new Promise(resolve => setTimeout(resolve, 5000));
                operation = await ai.operations.getVideosOperation({ operation });
            }

            if (operation.response?.generatedVideos?.[0]?.video?.uri) {
                const downloadLink = operation.response.generatedVideos[0].video.uri;
                const response = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                setGeneratedVideoUrl(url);
            } else {
                throw new Error("Video generation completed but no video URI was found.");
            }

        } catch (err: any) {
            console.error("Video generation error:", err);
            setError("An error occurred during video generation. Please try again.");
            if(err?.message?.includes('Requested entity was not found')) {
                 setError("Your API Key seems to be invalid. Please select a valid key.");
                 setApiKeySelected(false);
            }
        } finally {
            setIsLoading(false);
        }
    };
    
    if (isCheckingApiKey) {
        return <div className="h-full flex items-center justify-center"><Loader text="Checking API Key..." /></div>;
    }
    
    if (!apiKeySelected) {
        return (
            <div className="h-full flex flex-col items-center justify-center text-center p-4">
                <h2 className="text-xl font-bold mb-2">API Key Required</h2>
                <p className="max-w-md text-text-porcelain mb-4">Video generation with Veo requires you to select your own API key. Billing is associated with your Google Cloud project.</p>
                <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noopener noreferrer" className="text-accent-gold underline mb-6">Learn more about billing</a>
                <button onClick={handleSelectKey} className="px-6 py-2 bg-accent-gold text-bg-onyx rounded-lg font-semibold interactive-glow">Select API Key</button>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col md:flex-row">
            <aside className="w-full md:w-1/3 xl:w-1/4 p-4 border-b md:border-b-0 md:border-r border-border-subtle flex-shrink-0">
                 <h3 className="font-bold mb-4">Video Controls</h3>
                 <div className="space-y-6">
                     <div>
                        <label className="text-sm font-semibold text-text-porcelain">Aspect Ratio</label>
                        <div className="flex gap-2 mt-2">
                           <button onClick={() => setAspectRatio('16:9')} className={`p-2 w-full rounded-md text-sm transition-colors ${aspectRatio === '16:9' ? 'bg-accent-gold text-bg-onyx' : 'bg-bg-onyx hover:bg-bg-charcoal'}`}>Landscape</button>
                           <button onClick={() => setAspectRatio('9:16')} className={`p-2 w-full rounded-md text-sm transition-colors ${aspectRatio === '9:16' ? 'bg-accent-gold text-bg-onyx' : 'bg-bg-onyx hover:bg-bg-charcoal'}`}>Portrait</button>
                        </div>
                    </div>
                     <div>
                        <label className="text-sm font-semibold text-text-porcelain">Resolution</label>
                        <div className="flex gap-2 mt-2">
                           <button onClick={() => setResolution('720p')} className={`p-2 w-full rounded-md text-sm transition-colors ${resolution === '720p' ? 'bg-accent-gold text-bg-onyx' : 'bg-bg-onyx hover:bg-bg-charcoal'}`}>720p</button>
                           <button onClick={() => setResolution('1080p')} className={`p-2 w-full rounded-md text-sm transition-colors ${resolution === '1080p' ? 'bg-accent-gold text-bg-onyx' : 'bg-bg-onyx hover:bg-bg-charcoal'}`}>1080p</button>
                        </div>
                    </div>
                 </div>
            </aside>
             <main className="flex-1 p-4 flex flex-col">
                <div className="flex-1 min-h-0 bg-bg-onyx rounded-lg flex items-center justify-center border border-border-subtle relative">
                     {isLoading ? (
                        <div className="text-center p-4">
                            <Loader text={loadingMessage} />
                        </div>
                     ) : generatedVideoUrl ? (
                         <video src={generatedVideoUrl} controls autoPlay loop className="w-full h-full object-contain"></video>
                     ) : (
                         <p className="text-text-porcelain">Your generated video will appear here.</p>
                     )}
                </div>
                 <div className="flex-shrink-0 pt-4 space-y-2">
                     {generatedVideoUrl && <a href={generatedVideoUrl} download="generated-video.mp4" className="px-4 py-2 bg-bg-charcoal border border-border-subtle rounded-lg text-sm interactive-glow inline-block">Download Video</a>}
                     <div className="flex items-center space-x-2 glass-surface rounded-lg p-2">
                        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="A cinematic shot of..." className="flex-1 bg-transparent p-2 focus:outline-none resize-none" rows={2}/>
                        <button onClick={handleGenerate} disabled={isLoading} className="px-6 py-2 bg-accent-gold text-bg-onyx rounded-lg font-semibold interactive-glow disabled:opacity-50">{isLoading ? "..." : "Generate"}</button>
                     </div>
                     {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
                 </div>
             </main>
        </div>
    );
};


const MediaSuite: React.FC = () => {
    const [activeTab, setActiveTab] = useState<'image' | 'video'>('image');

    const TabButton: React.FC<{ tabName: 'image' | 'video', label: string }> = ({ tabName, label }) => (
        <button
            onClick={() => setActiveTab(tabName)}
            className={`px-4 py-2 text-sm font-semibold rounded-t-lg transition-colors relative ${activeTab === tabName ? 'text-accent-gold' : 'text-text-porcelain hover:text-white'}`}
        >
            {label}
            {activeTab === tabName && (
                <motion.div
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent-gold"
                    layoutId="media-tab-underline"
                    transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                />
            )}
        </button>
    );

    return (
        <div className="flex flex-col h-full bg-bg-charcoal/50">
            <header className="flex-shrink-0 border-b border-border-subtle px-4">
                <nav className="flex space-x-2">
                    <TabButton tabName="image" label="Image Studio" />
                    <TabButton tabName="video" label="Video Studio" />
                </nav>
            </header>
            <main className="flex-1 relative">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={activeTab}
                        className="w-full h-full absolute inset-0"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.2 }}
                    >
                        {activeTab === 'image' ? <ImageStudio /> : <VideoStudio />}
                    </motion.div>
                </AnimatePresence>
            </main>
        </div>
    );
};

const Scheduler: React.FC = () => (
    <div className="flex flex-col h-full p-4 items-center justify-center text-center">
        <h1 className="text-3xl font-bold text-text-platinum mb-4">Scheduler</h1>
        <p className="text-text-porcelain mt-2 max-w-md">The Scheduler is currently under development. Task automation and content planning are on the way.</p>
    </div>
);

// --- Persona Management Modal ---
const PersonaModal: React.FC<{
    isOpen: boolean;
    onClose: () => void;
    personas: Persona[];
    setPersonas: React.Dispatch<React.SetStateAction<Persona[]>>;
    onSelectPersona: (id: string) => void;
    activePersonaId: string;
}> = ({ isOpen, onClose, personas, setPersonas, onSelectPersona, activePersonaId }) => {
    const [view, setView] = useState<'list' | 'create'>('list');
    const [editingPersona, setEditingPersona] = useState<Persona | null>(null);

    // Create/Edit State
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [prompt, setPrompt] = useState('');
    const [voiceId, setVoiceId] = useState<Persona['voiceId']>('Kore');
    const [isGenerating, setIsGenerating] = useState(false);

    const handleGeneratePrompt = async () => {
        if (!description) return;
        setIsGenerating(true);
        setPrompt('');
        try {
            const ai = getAi();
            const metaPrompt = `Based on the following user description, generate a comprehensive and detailed system prompt for a conversational AI persona. The prompt must be structured to define the persona's core identity, personality, conversational style, and objectives. It should be directly usable as a system instruction for a large language model.\n\nUser's Persona Description: "${description}"\n\nThe generated prompt should include sections for:\n1. **Objective**: A clear, one-sentence mission for the AI.\n2. **Personality**: A list of key character traits and adjectives.\n3. **Tone**: Description of its voice and manner of speaking.\n4. **Conversational Framework**: Rules for how it interacts, such as asking questions, using empathy, or staying on topic.\n5. **Implicit Teaching**: What subtle communication skill it should model for the user.\n\nGenerate only the system prompt text, ready to be copied.`;
            
            const response = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: metaPrompt });
            setPrompt(response.text.trim());
        } catch (err) {
            console.error("Prompt generation error:", err);
            setPrompt("Sorry, I couldn't generate the prompt. Please try again.");
        } finally {
            setIsGenerating(false);
        }
    };
    
    const handleSave = () => {
        if (!name || !prompt) return;
        if (editingPersona) {
            setPersonas(personas.map(p => p.id === editingPersona.id ? { ...p, name, description, prompt, voiceId } : p));
        } else {
            const newPersona: Persona = {
                id: `custom-${Date.now()}`,
                name, description, prompt, voiceId, isCustom: true,
            };
            setPersonas([...personas, newPersona]);
        }
        resetAndClose();
    };

    const handleDelete = (id: string) => {
        setPersonas(personas.filter(p => p.id !== id));
    };

    const openCreator = (personaToEdit: Persona | null = null) => {
        setEditingPersona(personaToEdit);
        if (personaToEdit) {
            setName(personaToEdit.name);
            setDescription(personaToEdit.description);
            setPrompt(personaToEdit.prompt);
            setVoiceId(personaToEdit.voiceId);
        } else {
            setName('');
            setDescription('');
            setPrompt('');
            setVoiceId('Kore');
        }
        setView('create');
    };

    const resetAndClose = () => {
        setView('list');
        setEditingPersona(null);
        onClose();
    };
    
    const renderListView = () => (
        <>
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold">Select Persona</h3>
                <button onClick={onClose}>{ICONS.X_MARK}</button>
            </div>
            <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2">
                <div>
                    <h4 className="text-sm font-semibold text-text-porcelain mb-2">Default Personas</h4>
                    {personas.filter(p => !p.isCustom).map(p => (
                        <button key={p.id} onClick={() => onSelectPersona(p.id)} className={`w-full text-left p-3 mb-2 rounded-lg transition-colors ${activePersonaId === p.id ? 'bg-accent-gold/20' : 'hover:bg-accent-gold/10'}`}>
                            <p className={`font-semibold ${activePersonaId === p.id ? 'text-accent-gold' : ''}`}>{p.name}</p>
                            <p className="text-sm text-text-porcelain">{p.description}</p>
                        </button>
                    ))}
                </div>
                <div>
                    <h4 className="text-sm font-semibold text-text-porcelain mb-2">Custom Personas</h4>
                    {personas.filter(p => p.isCustom).map(p => (
                         <div key={p.id} className={`w-full text-left p-3 mb-2 rounded-lg transition-colors flex justify-between items-center ${activePersonaId === p.id ? 'bg-accent-gold/20' : 'hover:bg-accent-gold/10'}`}>
                            <button onClick={() => onSelectPersona(p.id)} className="flex-1 text-left">
                                <p className={`font-semibold ${activePersonaId === p.id ? 'text-accent-gold' : ''}`}>{p.name}</p>
                                <p className="text-sm text-text-porcelain">{p.description}</p>
                            </button>
                            <div className="flex items-center gap-2">
                                <button onClick={() => openCreator(p)} className="p-1 text-text-porcelain hover:text-white">{ICONS.PENCIL}</button>
                                <button onClick={() => handleDelete(p.id)} className="p-1 text-text-porcelain hover:text-red-500">{ICONS.TRASH}</button>
                            </div>
                        </div>
                    ))}
                    <button onClick={() => openCreator()} className="w-full flex items-center justify-center gap-2 p-3 mt-2 rounded-lg border-2 border-dashed border-border-subtle text-text-porcelain hover:bg-accent-gold/10 hover:border-accent-gold transition-colors">
                        {ICONS.USER_PLUS} Create New Persona
                    </button>
                </div>
            </div>
        </>
    );
    
    const renderCreateView = () => (
        <>
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-bold">{editingPersona ? 'Edit Persona' : 'Create Persona'}</h3>
                <button onClick={() => setView('list')}>{ICONS.CHEVRON_LEFT}</button>
            </div>
            <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2">
                <input type="text" placeholder="Persona Name (e.g., 'Creative Muse')" value={name} onChange={e => setName(e.target.value)} className="w-full p-2 bg-bg-onyx rounded-md border border-border-subtle focus:ring-1 focus:ring-accent-gold focus:outline-none" />
                <select value={voiceId} onChange={e => setVoiceId(e.target.value as Persona['voiceId'])} className="w-full p-2 bg-bg-onyx rounded-md border border-border-subtle focus:ring-1 focus:ring-accent-gold focus:outline-none">
                    {AVAILABLE_VOICES.map(v => <option key={v.id} value={v.id}>{v.name}</option>)}
                </select>
                <div>
                     <textarea placeholder="Describe your persona's core function (e.g., 'A supportive virtual girlfriend that helps me relax.')" value={description} onChange={e => setDescription(e.target.value)} rows={2} className="w-full p-2 bg-bg-onyx rounded-md border border-border-subtle focus:ring-1 focus:ring-accent-gold focus:outline-none resize-none" />
                     <button onClick={handleGeneratePrompt} disabled={isGenerating || !description} className="w-full mt-2 p-2 bg-bg-charcoal rounded-md text-sm font-semibold flex items-center justify-center gap-2 interactive-glow disabled:opacity-50">
                        {isGenerating ? <Loader text="Generating..."/> : <>{ICONS.SPARKLES} Generate Prompt with AI</>}
                    </button>
                </div>
                <textarea placeholder="The generated system prompt will appear here..." value={prompt} onChange={e => setPrompt(e.target.value)} rows={8} className="w-full p-2 bg-bg-onyx rounded-md border border-border-subtle focus:ring-1 focus:ring-accent-gold focus:outline-none" />
            </div>
            <div className="mt-6 flex justify-end gap-3">
                <button onClick={resetAndClose} className="px-4 py-2 bg-bg-charcoal rounded-lg interactive-glow">Cancel</button>
                <button onClick={handleSave} className="px-4 py-2 bg-accent-gold text-bg-onyx font-semibold rounded-lg interactive-glow">Save</button>
            </div>
        </>
    );

    return (
         <AnimatePresence>
            {isOpen && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 bg-black/60 flex items-center justify-center z-50">
                <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }} exit={{ scale: 0.95 }} className="w-full max-w-md glass-surface rounded-xl p-6">
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={view}
                            initial={{ opacity: 0, x: view === 'create' ? 20 : -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: view === 'create' ? -20 : 20 }}
                            transition={{ duration: 0.2 }}
                        >
                            {view === 'list' ? renderListView() : renderCreateView()}
                        </motion.div>
                    </AnimatePresence>
                </motion.div>
            </motion.div>
            )}
        </AnimatePresence>
    )
}

// --- Main App Component ---
const App: React.FC = () => {
    const [activeView, setActiveView] = useState<View>('chat');
    const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);
    const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
    const [personas, setPersonas] = useLocalStorage<Persona[]>('w3j-personas', defaultPersonas);
    const isMobile = useMediaQuery('(max-width: 768px)');
    
    const navItems = [
        { view: 'chat', label: 'Chat Agent', icon: ICONS.CHAT },
        { view: 'live', label: 'Live Agent', icon: ICONS.LIVE },
        { view: 'media', label: 'Media Suite', icon: ICONS.SPARKLES },
        { view: 'scheduler', label: 'Scheduler', icon: ICONS.CALENDAR },
    ];
    
    const handleNavClick = (view: View) => {
        setActiveView(view);
        if(isMobile) setIsMobileNavOpen(false);
    }
    
    const renderView = () => {
        switch(activeView) {
            case 'chat': return <ChatAgent personas={personas} setPersonas={setPersonas} />;
            case 'live': return <LiveAgent personas={personas} setPersonas={setPersonas} />;
            case 'media': return <MediaSuite />;
            case 'scheduler': return <Scheduler />;
            default: return <ChatAgent personas={personas} setPersonas={setPersonas} />;
        }
    };
    
    const Header = () => {
        const title = navItems.find(item => item.view === activeView)?.label || "W3J Power Suite";
        return (
             <header className="p-4 flex-shrink-0 flex items-center space-x-3 border-b border-border-subtle glass-surface md:hidden">
                 <button onClick={() => setIsMobileNavOpen(true)} className="p-1 text-text-platinum">{ICONS.MENU}</button>
                <h1 className="text-lg font-semibold text-text-platinum truncate">{title}</h1>
             </header>
        );
    }

    const pageVariants = {
      initial: { opacity: 0, filter: 'blur(4px)', scale: 0.98 },
      in: { opacity: 1, filter: 'blur(0px)', scale: 1 },
      out: { opacity: 0, filter: 'blur(4px)', scale: 1.02 },
    };

    const NavButton: React.FC<{ item: typeof navItems[0]; isCollapsed: boolean; }> = ({ item, isCollapsed }) => {
        const isActive = activeView === item.view;
        return (
            <motion.button 
                onClick={() => handleNavClick(item.view as View)} 
                className={`relative flex items-center gap-4 px-4 transition-colors duration-200 w-full h-14 rounded-lg
                    ${isCollapsed ? 'justify-center' : 'justify-start'}
                    ${isActive ? 'text-accent-gold' : 'text-text-porcelain hover:text-white'}`
                }
                title={item.label}
            >
                {isActive && (
                     <motion.div
                        className="absolute inset-0 bg-accent-gold/10 rounded-lg"
                        layoutId="active-nav-bg"
                        transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                    />
                )}
                 <div className="relative z-10">{item.icon}</div>
                <AnimatePresence>
                 {!isCollapsed && (
                    <motion.span 
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}
                        className="relative z-10 font-semibold whitespace-nowrap"
                    >
                        {item.label}
                    </motion.span>
                 )}
                </AnimatePresence>
            </motion.button>
        );
    };
    
    const MobileNavButton: React.FC<{ item: typeof navItems[0] }> = ({ item }) => (
         <motion.button 
            onClick={() => handleNavClick(item.view as View)} 
            className={`relative flex items-center justify-start gap-4 px-4 transition-colors duration-200 w-full h-16 text-lg ${activeView === item.view ? 'text-accent-gold' : 'text-text-porcelain hover:text-white'}`}
            title={item.label}
        >
            {item.icon}
            <span className="font-semibold">{item.label}</span>
        </motion.button>
    )

    const DesktopSidebar = () => (
        <motion.nav 
            className="h-full flex-shrink-0 flex flex-col items-center p-2 glass-surface"
            animate={{ width: isSidebarCollapsed ? '5rem' : '16rem' }}
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
        >
             <div className="h-16 w-full flex items-center justify-start px-4 space-x-3 text-white overflow-hidden">
                {ICONS.LOGO}
                {!isSidebarCollapsed && <span className="text-xl font-bold whitespace-nowrap">W3J Power Suite</span>}
             </div>
             <div className="flex-grow w-full flex flex-col items-center space-y-2">
                 {navItems.map(item => <NavButton key={item.view} item={item} isCollapsed={isSidebarCollapsed} />)}
             </div>
             <div className="w-full py-2">
                <button onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)} className="w-full h-12 flex items-center justify-center text-text-porcelain hover:text-accent-gold transition-colors rounded-lg hover:bg-white/5">
                    {isSidebarCollapsed ? ICONS.CHEVRON_RIGHT : ICONS.CHEVRON_LEFT}
                </button>
             </div>
        </motion.nav>
    );

    const MobileSidebar = () => (
         <AnimatePresence>
            {isMobileNavOpen && (
                <>
                    <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        onClick={() => setIsMobileNavOpen(false)}
                        className="absolute inset-0 bg-black/60 z-40"
                    />
                    <motion.nav 
                        className="absolute top-0 left-0 bottom-0 w-64 glass-surface z-50 flex flex-col p-4"
                        initial={{ x: '-100%' }}
                        animate={{ x: 0 }}
                        exit={{ x: '-100%' }}
                        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    >
                        <div className="h-16 w-full flex items-center justify-start space-x-3 text-white mb-4">
                            {ICONS.LOGO}
                            <span className="text-xl font-bold">W3J Power Suite</span>
                        </div>
                        <div className="flex-grow w-full flex flex-col items-center space-y-2">
                            {navItems.map(item => <MobileNavButton key={item.view} item={item} />)}
                        </div>
                        <div className="w-full"><MobileNavButton item={{ view: 'chat', label: 'Settings', icon: ICONS.SETTINGS }} /></div>
                    </motion.nav>
                </>
            )}
        </AnimatePresence>
    )

    return (
        <div className="h-screen w-screen p-0 md:p-8 flex items-center justify-center">
            <div className={`w-full h-full max-w-7xl flex relative overflow-hidden ${isMobile ? 'flex-col' : 'rounded-3xl glass-surface'}`}>
                {isMobile && <Header />}
                {!isMobile && <DesktopSidebar />}
                
                <main className="flex-1 relative">
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={activeView}
                            className="w-full h-full absolute inset-0"
                            variants={pageVariants}
                            initial="initial"
                            animate="in"
                            exit="out"
                            transition={{ duration: 0.3 }}
                        >
                            {renderView()}
                        </motion.div>
                    </AnimatePresence>
                </main>

                {isMobile && <MobileSidebar />}
            </div>
        </div>
    );
};

export default App;

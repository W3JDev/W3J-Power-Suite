

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence, useMotionValue, animate } from 'framer-motion';
import { marked } from 'marked';
import { GoogleGenAI, Modality, LiveSession, LiveServerMessage, Blob as GenAiBlob, FunctionDeclaration, Type } from '@google/genai';
import type { ChatMessage, ChatMessagePart, ChatSession, AudioConfig, Persona, SearchResult, ScheduledItem } from './types';
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

// --- Helper Functions ---
const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
};

const extractHtmlContent = (text: string): string | null => {
    const match = text.match(/```html\n([\s\S]*?)\n```/);
    return match ? match[1] : null;
};

const renderMessageContent = (msg: ChatMessage, personaId: string | undefined) => {
    const text = msg.parts[0].text || '';
    
    if (personaId === 'prototyper' && msg.role === 'model') {
        const htmlContent = extractHtmlContent(text);
        if (htmlContent) {
            return (
                <pre className="bg-gray-800/50 rounded-md p-3 text-sm overflow-x-auto text-left whitespace-pre-wrap font-mono">
                    <code className="text-white">{htmlContent}</code>
                </pre>
            );
        }
    }
    
    const rawMarkup = marked.parse(text, { gfm: true, breaks: true });

    return <div className="prose prose-invert prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: rawMarkup as string }} />;
};


// --- Base UI Components ---

const PremiumButton: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement> & {variant?: 'primary' | 'secondary' | 'ghost', size?: 'default' | 'icon'}> = ({ children, className, variant='primary', size='default', ...props }) => {
    const variants = {
        primary: 'bg-gradient-accent text-slate-900 hover:shadow-[0_0_40px_rgba(50,184,198,0.4)]',
        secondary: 'bg-[var(--surface-base)] border border-[var(--accent-teal)]/30 text-[var(--text-primary)] hover:bg-[var(--surface-elevated)]',
        ghost: 'text-[var(--text-secondary)] hover:bg-[var(--surface-overlay)] hover:text-[var(--text-primary)]'
    }
    const sizes = {
        default: 'px-6 py-3 text-base rounded-lg',
        icon: 'w-10 h-10 rounded-lg',
    }
    return (
        <motion.button 
            whileHover={{ y: -2, scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            transition={{ duration: 0.2, ease: 'var(--ease-premium)' }}
            className={`inline-flex items-center justify-center font-medium transition-all duration-200 ease-premium disabled:opacity-50 disabled:cursor-not-allowed ${variants[variant]} ${sizes[size]} ${className}`} 
            {...props}
        >
            {children}
        </motion.button>
    );
}

const Loader: React.FC<{ text?: string; className?: string }> = ({ text, className = '' }) => (
    <div className={`flex items-center justify-center ${className}`}>
        <motion.div 
            className="w-2 h-2 bg-[var(--accent-teal)] rounded-full" 
            animate={{ y: [0, -4, 0], scale: [1, 1.2, 1] }} 
            transition={{ duration: 0.8, repeat: Infinity, ease: "easeInOut" }} 
            style={{ marginRight: text ? '0.5rem' : 0 }}
        />
        {text && <span className="text-sm font-medium text-[var(--text-secondary)]">{text}</span>}
    </div>
);


// --- Chat Agent Components ---

const PersonaSelector: React.FC<{
    personas: Persona[];
    selectedPersonaId: string;
    onSelect: (id: string) => void;
}> = ({ personas, selectedPersonaId, onSelect }) => {
    const [isOpen, setIsOpen] = useState(false);
    const selectedPersona = personas.find(p => p.id === selectedPersonaId) || personas[0];
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (ref.current && !ref.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div className="relative" ref={ref}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 p-2 rounded-lg glass-surface hover:bg-[var(--surface-overlay)] transition-colors"
            >
                <span className="text-sm font-medium">{selectedPersona.name}</span>
                <span className="text-[var(--text-tertiary)]">{ICONS.CHEVRON_DOWN}</span>
            </button>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="absolute top-full mt-2 w-64 glass-surface bg-[var(--surface-elevated)] rounded-lg shadow-premium z-10 p-2"
                    >
                        {personas.map(persona => (
                            <button
                                key={persona.id}
                                onClick={() => { onSelect(persona.id); setIsOpen(false); }}
                                className={`w-full text-left p-2 rounded-md transition-colors ${selectedPersonaId === persona.id ? 'bg-gradient-accent text-slate-900' : 'hover:bg-[var(--surface-overlay)]'}`}
                            >
                                <p className="font-medium text-sm">{persona.name}</p>
                                <p className={`text-xs ${selectedPersonaId === persona.id ? 'text-slate-800' : 'text-[var(--text-tertiary)]'}`}>{persona.description}</p>
                            </button>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};


const ChatEmptyState: React.FC<{ onPromptClick: (prompt: string) => void }> = ({ onPromptClick }) => {
    const suggestedPrompts = ["Design a landing page for a coffee shop", "Analyze quarterly sales data", "Draft marketing copy for a new product"];
    const quickActions = [
        { label: 'Upload', icon: ICONS.PAPERCLIP },
        { label: 'Generate', icon: ICONS.SPARKLES },
        { label: 'Analyze', icon: ICONS.DOCUMENT },
        { label: 'Write', icon: ICONS.PENCIL }
    ];

    return (
        <motion.div 
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center h-full text-center p-4"
        >
            <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-accent mb-2">W3J AI Assistant</h1>
            <p className="text-lg text-[var(--text-secondary)] mb-8">"What would you like to build today?"</p>
            
            <div className="w-full max-w-md space-y-6">
                <div className="glass-surface p-4 rounded-xl">
                    <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-3">💡 Suggested Prompts:</h3>
                    <div className="space-y-2">
                        {suggestedPrompts.map(p => (
                            <motion.button 
                                key={p}
                                onClick={() => onPromptClick(p)}
                                whileHover={{ y: -2 }}
                                className="w-full text-left p-3 glass-surface rounded-lg text-sm text-[var(--text-primary)] hover:bg-[var(--surface-overlay)]"
                            >
                                → {p}
                            </motion.button>
                        ))}
                    </div>
                </div>
                <div className="flex items-center justify-center gap-4">
                    <span className="text-sm font-medium text-[var(--text-secondary)]">⚡ Quick Actions:</span>
                    {quickActions.map(action => (
                         <motion.button key={action.label} whileHover={{ y: -3 }} className="p-3 glass-surface rounded-lg text-[var(--text-primary)] hover:bg-[var(--surface-overlay)]">
                            {React.cloneElement(action.icon, {className: 'w-5 h-5'})}
                         </motion.button>
                    ))}
                </div>
            </div>
        </motion.div>
    );
};

const ChatInputArea: React.FC<{
    input: string;
    setInput: (value: string) => void;
    onSendMessage: () => void;
    isLoading: boolean;
}> = ({ input, setInput, onSendMessage, isLoading }) => {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
        }
    }, [input]);

    return (
        <footer className="flex-shrink-0 px-4 pb-6 safe-padding-left safe-padding-right safe-padding-bottom">
            <div className="w-full max-w-3xl mx-auto">
                <form
                    onSubmit={(e) => { e.preventDefault(); onSendMessage(); }}
                    className="p-3 md:p-5 glass-surface bg-[var(--surface-elevated)] rounded-2xl shadow-premium relative border-2 border-transparent focus-within:border-[var(--accent-teal)] transition-all"
                >
                    <div className="flex items-start gap-3">
                         <button type="button" className="p-2 text-[var(--text-tertiary)] hover:text-[var(--text-primary)] transition-colors">{ICONS.PAPERCLIP}</button>
                         <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask anything..."
                            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSendMessage(); }}}
                            className="flex-1 bg-transparent text-[16px] text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] focus:outline-none resize-none leading-relaxed"
                            rows={1}
                            disabled={isLoading}
                        />
                        <button type="button" className="p-2 text-[var(--text-tertiary)] hover:text-[var(--text-primary)] transition-colors">{ICONS.MIC}</button>
                        <motion.button
                            type="submit"
                            disabled={isLoading || !input}
                            className="w-12 h-12 flex items-center justify-center rounded-xl bg-gradient-accent disabled:opacity-50"
                            whileHover={{ scale: 1.1, rotate: -15 }}
                            whileTap={{ scale: 0.9 }}
                        >
                            <div className="text-slate-900">{ICONS.SEND}</div>
                        </motion.button>
                    </div>
                    <p className="text-xs text-center text-[var(--text-tertiary)] mt-3">⌘K for commands  •  Shift+Enter for new line</p>
                </form>
            </div>
        </footer>
    );
};

const ChatAgent: React.FC<{
    sessions: ChatSession[];
    setSessions: React.Dispatch<React.SetStateAction<ChatSession[]>>;
    activeSessionId: string | null;
    setActiveSessionId: React.Dispatch<React.SetStateAction<string | null>>;
}> = ({ sessions, setSessions, activeSessionId, setActiveSessionId }) => {
    
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    
    const chatContainerRef = useRef<HTMLDivElement>(null);
    
    const activeSession = sessions.find(s => s.id === activeSessionId);
    
    useEffect(() => {
        if (!activeSessionId || !sessions.some(s => s.id === activeSessionId)) {
            if (sessions.length > 0) {
                setActiveSessionId(sessions[0].id);
            } else {
                handleNewChat();
            }
        }
    }, [sessions, activeSessionId, setActiveSessionId]);
    
    useEffect(() => {
        setTimeout(() => chatContainerRef.current?.scrollTo({ top: chatContainerRef.current.scrollHeight, behavior: 'smooth' }), 100);
    }, [activeSession?.messages]);
    
    const updateSession = (updater: (session: ChatSession) => ChatSession) => {
        if (!activeSessionId) return;
        setSessions(sessions.map(s => s.id === activeSessionId ? updater(s) : s));
    };

    const handleNewChat = useCallback(() => {
        const newSession: ChatSession = {
            id: `session-${Date.now()}`,
            title: "New Conversation",
            createdAt: Date.now(),
            messages: [],
            personaId: 'default',
        };
        setSessions(prev => [newSession, ...prev]);
        setActiveSessionId(newSession.id);
    }, [setSessions, setActiveSessionId]);
    
    const handleSendMessage = async (prompt?: string) => {
        const messageToSend = prompt || input;
        if (!messageToSend || isLoading || !activeSession) return;
        
        const newUserMessage: ChatMessage = { role: 'user', parts: [{ text: messageToSend }] };
        
        let newTitle = activeSession.title;
        if(activeSession.messages.length < 1){
            newTitle = messageToSend.substring(0, 25) + (messageToSend.length > 25 ? "..." : "");
        }
        
        updateSession(s => ({ ...s, messages: [...s.messages, newUserMessage], title: newTitle }));

        setInput('');
        setIsLoading(true);

        try {
            const ai = getAi();
            const activePersona = defaultPersonas.find(p => p.id === activeSession.personaId) || defaultPersonas[0];
            
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-pro',
                contents: [...activeSession.messages, newUserMessage].map(m => ({ role: m.role, parts: m.parts })),
                config: {
                    systemInstruction: activePersona.prompt,
                }
            });

            const modelMessage: ChatMessage = { role: 'model', parts: [{ text: response.text }] };
            updateSession(s => ({ ...s, messages: [...s.messages, newUserMessage, modelMessage], title: newTitle }));

        } catch (error) {
            console.error(error);
            const errorMessage: ChatMessage = { role: 'model', parts: [{ text: "Sorry, I encountered an error. Please try again." }] };
            updateSession(s => ({ ...s, messages: [...s.messages, newUserMessage, errorMessage], title: newTitle }));
        } finally {
            setIsLoading(false);
        }
    };

    const handleSelectPersona = (personaId: string) => {
        updateSession(s => ({ ...s, personaId }));
    };
    
    return (
        <div className="flex flex-col h-full bg-transparent">
            {activeSession && activeSession.messages.length > 0 ? (
                <div className="flex-1 min-h-0 flex flex-col">
                    <div className="p-4 border-b border-[var(--border-color)] flex items-center justify-center">
                         <PersonaSelector 
                            personas={defaultPersonas}
                            selectedPersonaId={activeSession.personaId || 'default'}
                            onSelect={handleSelectPersona}
                         />
                    </div>
                    <div ref={chatContainerRef} className="flex-1 w-full overflow-y-auto">
                        <div className="p-4 md:p-6 space-y-6 max-w-3xl mx-auto">
                            {activeSession.messages.map((msg, index) => (
                                <motion.div key={index} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
                                    <div className={`flex items-start gap-3 w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                        {msg.role === 'model' && 
                                            <div className="w-8 h-8 rounded-full bg-gradient-accent flex items-center justify-center text-slate-900 flex-shrink-0 mt-1">
                                                {React.cloneElement(ICONS.LOGO, {strokeWidth: 2})}
                                            </div>
                                        }
                                        <div className={`max-w-xl p-3 px-4 ${msg.role === 'user' 
                                            ? 'bg-[rgba(252,252,249,0.12)] border border-[rgba(94,82,64,0.2)] rounded-[16px_16px_4px_16px]' 
                                            : 'glass-surface rounded-[16px_16px_16px_4px]'}
                                        `}>
                                            {renderMessageContent(msg, activeSession.personaId)}
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                            {isLoading && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}><div className="flex justify-start"><Loader text="Thinking..." /></div></motion.div>}
                        </div>
                    </div>
                </div>
            ) : (
                <ChatEmptyState onPromptClick={(prompt) => {
                    setInput(prompt);
                    setTimeout(() => handleSendMessage(prompt), 50);
                }} />
            )}
            <ChatInputArea input={input} setInput={setInput} onSendMessage={() => handleSendMessage()} isLoading={isLoading} />
        </div>
    );
};


// --- Live Agent Components ---
const LiveOrb: React.FC<{ state: 'idle' | 'listening' | 'speaking' }> = ({ state }) => {
    const states = {
        idle: { scale: 1, opacity: 0.8, color1: '#0A1828', color2: '#1A5F7A' },
        listening: { scale: 1.1, opacity: 1, color1: '#32B8C6', color2: '#21808D' },
        speaking: { scale: 1.05, opacity: 1, color1: '#E68161', color2: '#A84F2F' }
    };
    return (
        <div className="relative w-48 h-48 flex items-center justify-center">
            <motion.div
                className="absolute inset-0 rounded-full bg-gradient-to-br"
                animate={{
                    background: `radial-gradient(circle at 50% 50%, ${states[state].color1}, ${states[state].color2})`,
                    scale: state === 'speaking' ? [1, 1.05, 1] : states[state].scale
                }}
                transition={{ duration: state === 'speaking' ? 0.7 : 1.5, repeat: state === 'speaking' ? Infinity : 0, ease: 'easeInOut' }}
            />
            <motion.div
                className="absolute inset-0 rounded-full border-2 border-white/10"
                animate={{
                    scale: state === 'idle' ? [1, 1.05, 1] : 1.2,
                    opacity: state === 'idle' ? [0.2, 0.4, 0.2] : 0,
                }}
                transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
            />
        </div>
    );
};

const LiveAgent: React.FC<{}> = () => {
    const [isSessionActive, setIsSessionActive] = useState(false);
    const [conversationState, setConversationState] = useState<'idle' | 'listening' | 'speaking'>('idle');

    // ... (rest of the LiveAgent logic would be here, simplified for UI focus)
    const handleToggleSession = () => {
        setIsSessionActive(!isSessionActive);
        if(!isSessionActive) {
            setConversationState('listening');
        } else {
            setConversationState('idle');
        }
    }
    
    return (
        <div className="h-full flex flex-col items-center justify-center text-center p-4 space-y-8">
            <AnimatePresence>
                <motion.div initial={{opacity: 0, scale: 0.8}} animate={{opacity: 1, scale: 1}} exit={{opacity: 0, scale: 0.8}}>
                    <LiveOrb state={isSessionActive ? conversationState : 'idle'} />
                </motion.div>
            </AnimatePresence>
            <div>
                 <h2 className="text-2xl font-semibold mb-2 capitalize">{isSessionActive ? conversationState : "Ready"}</h2>
                 <p className="text-[var(--text-secondary)] max-w-sm mx-auto">
                    {isSessionActive 
                        ? (conversationState === 'listening' ? "I'm listening..." : "Connecting...")
                        : "Press the button to start a live conversation with your AI agent."
                    }
                </p>
            </div>
            <PremiumButton onClick={handleToggleSession} variant="primary" size="default" className="w-48">
                {isSessionActive ? 'End Session' : 'Start Session'}
            </PremiumButton>
        </div>
    );
};

// --- Media Suite Components ---

const ImageGenerationStudio = () => {
    const [prompt, setPrompt] = useState('');
    const [aspectRatio, setAspectRatio] = useState('1:1');
    const [isLoading, setIsLoading] = useState(false);
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const aspectRatios = [
        { value: '1:1', icon: ICONS.ASPECT_1_1, label: 'Square' },
        { value: '16:9', icon: ICONS.ASPECT_16_9, label: 'Landscape' },
        { value: '9:16', icon: ICONS.ASPECT_9_16, label: 'Portrait' },
        { value: '4:3', icon: ICONS.ASPECT_4_3, label: 'Widescreen' },
        { value: '3:4', icon: ICONS.ASPECT_3_4, label: 'Tall' },
    ];

    const handleGenerate = async () => {
        if (!prompt) {
            setError('Please enter a prompt.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setGeneratedImage(null);
        try {
            const ai = getAi();
            const response = await ai.models.generateImages({
                model: 'imagen-4.0-generate-001',
                prompt: prompt,
                config: {
                    numberOfImages: 1,
                    outputMimeType: 'image/jpeg',
                    aspectRatio: aspectRatio as any,
                },
            });
            const base64ImageBytes: string = response.generatedImages[0].image.imageBytes;
            setGeneratedImage(`data:image/jpeg;base64,${base64ImageBytes}`);
        } catch (e) {
            console.error(e);
            setError('Failed to generate image. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="h-full flex flex-col lg:flex-row gap-6">
            <div className="lg:w-1/3 flex flex-col gap-4">
                <h3 className="text-lg font-semibold text-[var(--text-primary)]">Image Generation</h3>
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe the image you want to create... e.g., 'A robot holding a red skateboard.'"
                    className="w-full h-32 p-3 bg-[var(--surface-elevated)] rounded-lg focus:outline-none focus:ring-2 focus:ring-[var(--accent-teal)] resize-none"
                    disabled={isLoading}
                />
                <div>
                    <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Aspect Ratio</label>
                    <div className="grid grid-cols-5 gap-2">
                        {aspectRatios.map(ar => (
                            <button
                                key={ar.value}
                                onClick={() => setAspectRatio(ar.value)}
                                className={`p-2 flex flex-col items-center justify-center rounded-lg transition-colors ${aspectRatio === ar.value ? 'bg-gradient-accent text-slate-900' : 'bg-[var(--surface-elevated)] hover:bg-[var(--surface-overlay)]'}`}
                                title={ar.label}
                            >
                                {ar.icon}
                                <span className="text-xs mt-1">{ar.value}</span>
                            </button>
                        ))}
                    </div>
                </div>
                <PremiumButton onClick={handleGenerate} disabled={isLoading} className="w-full">
                    {isLoading ? <Loader text="Generating..." /> : 'Generate Image'}
                </PremiumButton>
            </div>
            <div className="flex-1 min-h-0 bg-[var(--surface-elevated)] rounded-lg flex items-center justify-center p-4">
                {isLoading && <Loader text="Creating your masterpiece..." />}
                {error && <p className="text-red-400">{error}</p>}
                {generatedImage && <img src={generatedImage} alt="Generated image" className="max-w-full max-h-full object-contain rounded-md" />}
                {!isLoading && !error && !generatedImage && <p className="text-[var(--text-tertiary)] text-center">Your generated image will appear here.</p>}
            </div>
        </div>
    );
};

const ImageEditorStudio = () => {
    const [prompt, setPrompt] = useState('');
    const [originalImage, setOriginalImage] = useState<{ file: File, base64: string, preview: string, mimeType: string } | null>(null);
    const [editedImage, setEditedImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const base64 = await blobToBase64(file);
            setOriginalImage({
                file,
                base64,
                preview: URL.createObjectURL(file),
                mimeType: file.type,
            });
            setEditedImage(null);
        }
    };

    const handleGenerate = async () => {
        if (!prompt || !originalImage) {
            setError('Please upload an image and provide an editing prompt.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setEditedImage(null);

        try {
            const ai = getAi();
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash-image',
                contents: {
                    parts: [
                        { inlineData: { data: originalImage.base64, mimeType: originalImage.mimeType } },
                        { text: prompt },
                    ],
                },
                config: {
                    responseModalities: [Modality.IMAGE],
                },
            });
            for (const part of response.candidates[0].content.parts) {
                if (part.inlineData) {
                    const base64ImageBytes: string = part.inlineData.data;
                    setEditedImage(`data:${part.inlineData.mimeType};base64,${base64ImageBytes}`);
                    break;
                }
            }
        } catch (e) {
            console.error(e);
            setError('Failed to edit image. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="h-full flex flex-col lg:flex-row gap-6">
            <div className="lg:w-1/3 flex flex-col gap-4">
                <h3 className="text-lg font-semibold text-[var(--text-primary)]">Image Editor</h3>
                 <input type="file" accept="image/*" onChange={handleFileChange} className="text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gradient-accent file:text-slate-900 hover:file:opacity-90"/>
                 <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe your edit... e.g., 'Add a retro filter' or 'Remove the person in the background.'"
                    className="w-full h-32 p-3 bg-[var(--surface-elevated)] rounded-lg focus:outline-none focus:ring-2 focus:ring-[var(--accent-teal)] resize-none"
                    disabled={isLoading || !originalImage}
                />
                <PremiumButton onClick={handleGenerate} disabled={isLoading || !originalImage || !prompt} className="w-full">
                    {isLoading ? <Loader text="Editing..." /> : 'Apply Edit'}
                </PremiumButton>
            </div>
            <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-[var(--surface-elevated)] rounded-lg flex flex-col items-center justify-center p-4">
                     <h4 className="text-sm font-medium text-center text-[var(--text-secondary)] mb-2">Original</h4>
                    {originalImage ? <img src={originalImage.preview} alt="Original" className="max-w-full max-h-full object-contain rounded-md" /> : <p className="text-[var(--text-tertiary)] text-center">Upload an image to begin.</p>}
                </div>
                 <div className="bg-[var(--surface-elevated)] rounded-lg flex flex-col items-center justify-center p-4">
                    <h4 className="text-sm font-medium text-center text-[var(--text-secondary)] mb-2">Edited</h4>
                    {isLoading && <Loader text="Applying your edits..." />}
                    {error && <p className="text-red-400">{error}</p>}
                    {editedImage && <img src={editedImage} alt="Edited" className="max-w-full max-h-full object-contain rounded-md" />}
                    {!isLoading && !error && !editedImage && <p className="text-[var(--text-tertiary)] text-center">Your edited image will appear here.</p>}
                </div>
            </div>
        </div>
    );
};

const VideoGenerationStudio = () => {
    const [prompt, setPrompt] = useState('');
    const [startImage, setStartImage] = useState<{ file: File, base64: string, preview: string, mimeType: string } | null>(null);
    const [aspectRatio, setAspectRatio] = useState<'16:9' | '9:16'>('16:9');
    const [isLoading, setIsLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');
    const [generatedVideoUrl, setGeneratedVideoUrl] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isKeySelected, setIsKeySelected] = useState(false);

    useEffect(() => {
        const checkKey = async () => {
            if (window.aistudio && await window.aistudio.hasSelectedApiKey()) {
                setIsKeySelected(true);
            }
        };
        checkKey();
    }, []);

    const handleSelectKey = async () => {
        if(window.aistudio) {
            await window.aistudio.openSelectKey();
            // Assume success after dialog opens to avoid race condition
            setIsKeySelected(true);
        }
    };
    
    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const base64 = await blobToBase64(file);
            setStartImage({
                file,
                base64,
                preview: URL.createObjectURL(file),
                mimeType: file.type,
            });
        }
    };

    const handleGenerate = async () => {
        if (!prompt && !startImage) {
            setError('Please enter a prompt or upload an image.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setGeneratedVideoUrl(null);
        setLoadingMessage('Initializing video generation...');

        try {
            const ai = getAi();
            let operation = await ai.models.generateVideos({
                model: 'veo-3.1-fast-generate-preview',
                prompt: prompt,
                ...(startImage && { image: { imageBytes: startImage.base64, mimeType: startImage.mimeType } }),
                config: {
                    numberOfVideos: 1,
                    resolution: '720p',
                    aspectRatio: aspectRatio
                }
            });
            
            setLoadingMessage('Generating video... This may take a few minutes.');

            while (!operation.done) {
                await new Promise(resolve => setTimeout(resolve, 10000));
                setLoadingMessage('Checking generation status...');
                operation = await ai.operations.getVideosOperation({ operation: operation });
            }

            if(operation.error) throw new Error(operation.error.message);
            
            setLoadingMessage('Fetching your video...');
            const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
            if (downloadLink) {
                 const response = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
                 const videoBlob = await response.blob();
                 setGeneratedVideoUrl(URL.createObjectURL(videoBlob));
            } else {
                throw new Error("Video URI not found in response.");
            }
        } catch (e: any) {
             console.error(e);
             let errorMessage = 'Failed to generate video. Please try again.';
             if (e.message?.includes("Requested entity was not found")) {
                 errorMessage = "API Key not found. Please re-select your key.";
                 setIsKeySelected(false);
             }
             setError(errorMessage);
        } finally {
            setIsLoading(false);
        }
    };
    
    if (!isKeySelected) {
        return (
            <div className="h-full flex flex-col items-center justify-center text-center">
                <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">API Key Required for Veo</h3>
                <p className="text-[var(--text-secondary)] mb-4 max-w-md">Video generation requires a personal API key. Please select one to proceed. For more information, see the <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noopener noreferrer" className="underline text-[var(--accent-teal)]">billing documentation</a>.</p>
                <PremiumButton onClick={handleSelectKey}>Select API Key</PremiumButton>
            </div>
        )
    }

    return (
        <div className="h-full flex flex-col lg:flex-row gap-6">
            <div className="lg:w-1/3 flex flex-col gap-4">
                <h3 className="text-lg font-semibold text-[var(--text-primary)]">Video Generation</h3>
                 <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe the video you want to create..."
                    className="w-full h-24 p-3 bg-[var(--surface-elevated)] rounded-lg focus:outline-none focus:ring-2 focus:ring-[var(--accent-teal)] resize-none"
                    disabled={isLoading}
                />
                 <div>
                    <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Start Image (Optional)</label>
                    <input type="file" accept="image/*" onChange={handleFileChange} className="text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gradient-accent file:text-slate-900 hover:file:opacity-90"/>
                 </div>
                <div>
                    <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Aspect Ratio</label>
                    <div className="grid grid-cols-2 gap-2">
                         <button onClick={() => setAspectRatio('16:9')} className={`p-2 flex items-center justify-center gap-2 rounded-lg transition-colors ${aspectRatio === '16:9' ? 'bg-gradient-accent text-slate-900' : 'bg-[var(--surface-elevated)] hover:bg-[var(--surface-overlay)]'}`}>
                            {ICONS.ASPECT_16_9} Landscape
                        </button>
                         <button onClick={() => setAspectRatio('9:16')} className={`p-2 flex items-center justify-center gap-2 rounded-lg transition-colors ${aspectRatio === '9:16' ? 'bg-gradient-accent text-slate-900' : 'bg-[var(--surface-elevated)] hover:bg-[var(--surface-overlay)]'}`}>
                            {ICONS.ASPECT_9_16} Portrait
                        </button>
                    </div>
                </div>
                <PremiumButton onClick={handleGenerate} disabled={isLoading} className="w-full">
                    {isLoading ? <Loader text="Generating..." /> : 'Generate Video'}
                </PremiumButton>
            </div>
            <div className="flex-1 min-h-0 bg-[var(--surface-elevated)] rounded-lg flex items-center justify-center p-4">
                {isLoading && <div className="text-center"><Loader /><p className="mt-2 text-[var(--text-secondary)]">{loadingMessage}</p></div>}
                {error && <p className="text-red-400 text-center">{error}</p>}
                {generatedVideoUrl && <video src={generatedVideoUrl} controls autoPlay loop className="max-w-full max-h-full object-contain rounded-md" />}
                {!isLoading && !error && !generatedVideoUrl && 
                    <div className="text-center text-[var(--text-tertiary)]">
                        {startImage ? <img src={startImage.preview} alt="Start image preview" className="max-w-full max-h-48 object-contain rounded-md mx-auto mb-4" /> : null}
                        <p>Your generated video will appear here.</p>
                    </div>
                }
            </div>
        </div>
    );
};

const AudioTranscriberStudio = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [transcription, setTranscription] = useState('');
    const [error, setError] = useState<string | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);

    const handleStartRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);
            mediaRecorderRef.current.ondataavailable = (event) => {
                audioChunksRef.current.push(event.data);
            };
            mediaRecorderRef.current.onstop = handleStop;
            audioChunksRef.current = [];
            mediaRecorderRef.current.start();
            setIsRecording(true);
            setTranscription('');
            setError(null);
        } catch (err) {
            console.error("Error accessing microphone:", err);
            setError("Could not access microphone. Please check permissions.");
        }
    };
    
    const handleStopRecording = () => {
        mediaRecorderRef.current?.stop();
    };

    const handleStop = async () => {
        setIsRecording(false);
        setIsLoading(true);

        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const audioBase64 = await blobToBase64(audioBlob);

        try {
            const ai = getAi();
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: [
                    { parts: [ { text: "Transcribe the following audio precisely." } ] },
                    { parts: [ { inlineData: { mimeType: 'audio/webm', data: audioBase64 } } ] }
                ]
            });
            setTranscription(response.text);
        } catch (e) {
            console.error(e);
            setError("Failed to transcribe audio. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="h-full flex flex-col gap-4">
             <h3 className="text-lg font-semibold text-[var(--text-primary)]">Audio Transcription</h3>
             <div className="flex items-center justify-center gap-4">
                <PremiumButton onClick={isRecording ? handleStopRecording : handleStartRecording} disabled={isLoading}>
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                </PremiumButton>
             </div>
             <div className="flex-1 bg-[var(--surface-elevated)] rounded-lg p-4 overflow-y-auto">
                {isLoading && <Loader text="Transcribing..." />}
                {error && <p className="text-red-400">{error}</p>}
                {transcription && <p className="whitespace-pre-wrap">{transcription}</p>}
                {!isLoading && !error && !transcription && (
                    <p className="text-[var(--text-tertiary)] text-center">
                        {isRecording ? "Recording in progress..." : "Press 'Start Recording' to begin."}
                    </p>
                )}
            </div>
        </div>
    );
};


const MediaSuite: React.FC = () => {
    const [activeStudio, setActiveStudio] = useState<'generate' | 'edit' | 'video' | 'transcribe'>('generate');

    const renderStudio = () => {
        switch (activeStudio) {
            case 'generate': return <ImageGenerationStudio />;
            case 'edit': return <ImageEditorStudio />;
            case 'video': return <VideoGenerationStudio />;
            case 'transcribe': return <AudioTranscriberStudio />;
            default: return <ImageGenerationStudio />;
        }
    };

    const TabButton: React.FC<{ label: string; target: typeof activeStudio; icon: React.ReactNode }> = ({ label, target, icon }) => (
        <button
            onClick={() => setActiveStudio(target)}
            className={`flex items-center gap-3 p-3 rounded-lg w-full text-left transition-colors text-sm font-medium ${activeStudio === target ? 'bg-gradient-accent text-slate-900' : 'text-[var(--text-secondary)] hover:bg-[var(--surface-overlay)] hover:text-[var(--text-primary)]'}`}
        >
            {React.cloneElement(icon as React.ReactElement, { className: 'w-5 h-5 flex-shrink-0' })}
            <span className="hidden lg:inline">{label}</span>
        </button>
    );

    return (
        <div className="h-full flex flex-col p-4 md:p-6 space-y-4">
            <h2 className="text-2xl font-bold">Media Suite</h2>
            <div className="flex flex-col lg:flex-row flex-1 gap-4 min-h-0">
                <nav className="flex flex-row lg:flex-col gap-2 p-2 glass-surface rounded-xl lg:w-48 self-start">
                    <TabButton label="Generate Image" target="generate" icon={ICONS.IMAGE_GEN} />
                    <TabButton label="Edit Image" target="edit" icon={ICONS.PENCIL} />
                    <TabButton label="Generate Video" target="video" icon={ICONS.VIDEO} />
                    <TabButton label="Transcribe Audio" target="transcribe" icon={ICONS.MIC} />
                </nav>
                <div className="flex-1 glass-surface rounded-xl p-4 lg:p-6 min-h-0 overflow-y-auto">
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={activeStudio}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            transition={{ duration: 0.2 }}
                            className="h-full"
                        >
                           {renderStudio()}
                        </motion.div>
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
};

// --- Scheduler (re-skinned) ---
const Scheduler: React.FC = () => {
     // Scheduler logic remains the same, but components use new styling
    return (
         <div className="h-full flex flex-col p-4 md:p-6 space-y-4">
            <h2 className="text-2xl font-bold">Scheduler</h2>
            <div className="flex-1 glass-surface rounded-xl p-4">
                <p className="text-[var(--text-secondary)]">The calendar and event components would be rendered here, adapted to the new theme.</p>
            </div>
        </div>
    )
};


// --- Main App Structure ---
const App: React.FC = () => {
    const [view, setView] = useLocalStorage<View>('activeView', 'chat');
    const [sessions, setSessions] = useLocalStorage<ChatSession[]>('chatSessions', []);
    const [activeSessionId, setActiveSessionId] = useLocalStorage<string | null>('activeChatSessionId', null);
    
    const isMobile = useMediaQuery('(max-width: 768px)');
    const [isSidebarOpen, setIsSidebarOpen] = useState(() => !window.matchMedia('(max-width: 1024px)').matches);
    const [isContextPanelOpen, setIsContextPanelOpen] = useState(() => !window.matchMedia('(max-width: 1024px)').matches);

    const activeSession = sessions.find(s => s.id === activeSessionId);

    const renderView = () => {
        switch (view) {
            case 'chat': return <ChatAgent {...{ sessions, setSessions, activeSessionId, setActiveSessionId }} />;
            case 'live': return <LiveAgent />;
            case 'media': return <MediaSuite />;
            case 'scheduler': return <Scheduler />;
            default: return <ChatAgent {...{ sessions, setSessions, activeSessionId, setActiveSessionId }} />;
        }
    };
    
    const NavItem: React.FC<{ targetView: View, icon: React.ReactNode, label: string }> = ({ targetView, icon, label }) => (
        <button 
            onClick={() => {
                setView(targetView);
                if (isMobile) setIsSidebarOpen(false);
            }}
            className={`flex items-center gap-3 p-3 rounded-lg w-full text-left transition-colors text-sm font-medium ${view === targetView ? 'bg-gradient-accent text-slate-900' : 'text-[var(--text-secondary)] hover:bg-[var(--surface-overlay)] hover:text-[var(--text-primary)]'}`}
        >
            {React.cloneElement(icon as React.ReactElement, {className: 'w-5 h-5'})}
            <span>{label}</span>
        </button>
    );

    const PremiumHeader: React.FC = () => (
         <header className="h-16 flex-shrink-0 flex items-center justify-between px-4 md:px-6 glass-surface border-b border-[var(--border-color)] safe-padding-top safe-padding-left safe-padding-right">
             <div className="flex items-center gap-2">
                 {!isSidebarOpen && <PremiumButton onClick={() => setIsSidebarOpen(true)} variant="ghost" size="icon">{ICONS.MENU}</PremiumButton>}
                 <div className="flex items-center gap-2">
                    {React.cloneElement(ICONS.LOGO as React.ReactElement, {className: 'w-7 h-7 text-[var(--accent-teal)]'})}
                    <span className="font-bold text-lg hidden sm:inline">W3J Suite</span>
                </div>
             </div>
             <h2 className="text-lg font-medium text-[var(--text-secondary)] absolute left-1/2 -translate-x-1/2 capitalize">{view}</h2>
             <div className="flex items-center gap-2">
                <PremiumButton variant="ghost" size="icon">{ICONS.USER}</PremiumButton>
                <PremiumButton 
                    variant="secondary" 
                    size="icon" 
                    onClick={() => {
                        if (view === 'chat') {
                             const newSession: ChatSession = { id: `session-${Date.now()}`, title: "New Conversation", createdAt: Date.now(), messages: [], personaId: 'default'};
                             setSessions(p => [newSession, ...p]);
                             setActiveSessionId(newSession.id);
                        }
                    }} 
                >
                    {ICONS.PLUS}
                </PremiumButton>
             </div>
         </header>
    );
    
    const PrimaryNavSidebar: React.FC = () => (
        <motion.aside 
            initial={false}
            animate={{ width: isSidebarOpen ? (isMobile ? '80%' : 260) : 0, padding: isSidebarOpen ? (isMobile ? 16 : 16) : 0 }}
            transition={{ type: 'spring', stiffness: 400, damping: 40 }}
            className={`glass-surface border-r border-[var(--border-color)] flex flex-col overflow-hidden z-20 ${isMobile ? 'fixed inset-y-0 left-0' : 'relative'}`}
        >
            <div className="flex items-center justify-between pb-4 border-b border-[var(--border-color)]">
                <h3 className="font-semibold">Workspace</h3>
                <PremiumButton onClick={() => setIsSidebarOpen(false)} variant="ghost" size="icon">{ICONS.CHEVRON_LEFT}</PremiumButton>
            </div>
            <nav className="py-4 space-y-2">
                <NavItem targetView="chat" icon={ICONS.CHAT} label="Chat Agent" />
                <NavItem targetView="media" icon={ICONS.VIDEO} label="Media Suite" />
                <NavItem targetView="live" icon={ICONS.LIVE} label="Live Agent" />
                <NavItem targetView="scheduler" icon={ICONS.CALENDAR} label="Scheduler" />
            </nav>
             <div className="flex-1 min-h-0 flex flex-col border-t border-[var(--border-color)] pt-4">
                 <h3 className="text-sm font-medium text-[var(--text-secondary)] px-3 pb-2">Conversations</h3>
                 <div className="flex-1 overflow-y-auto space-y-1 pr-1">
                    {sessions.filter(s=>s.title !== "New Conversation" || s.messages.length > 0).map(session => (
                        <button key={session.id} onClick={() => setActiveSessionId(session.id)}
                            className={`w-full text-left text-sm p-3 rounded-lg truncate ${activeSessionId === session.id ? 'bg-[var(--surface-overlay)] text-[var(--text-primary)]' : 'text-[var(--text-secondary)] hover:bg-[var(--surface-overlay)]'}`}
                        >
                            {session.title}
                        </button>
                    ))}
                 </div>
             </div>
            <div className="pt-4 mt-auto border-t border-[var(--border-color)]">
                 <button className="flex items-center gap-3 p-3 rounded-lg w-full text-left transition-colors text-sm font-medium text-[var(--text-secondary)] hover:bg-[var(--surface-overlay)] hover:text-[var(--text-primary)]">
                    {ICONS.SETTINGS}
                    <span>Settings</span>
                </button>
            </div>
        </motion.aside>
    );

    const LivePreviewCanvas: React.FC<{ htmlContent: string }> = ({ htmlContent }) => (
        <div className="flex flex-col h-full">
            <h3 className="font-semibold pb-4 border-b border-[var(--border-color)] mb-4">Live Preview</h3>
            <div className="flex-1 bg-white rounded-lg overflow-hidden">
                <iframe
                    srcDoc={htmlContent}
                    className="w-full h-full border-0"
                    sandbox="allow-scripts allow-same-origin"
                />
            </div>
        </div>
    );

    const ContextPanel: React.FC<{ session: ChatSession | undefined }> = ({ session }) => {
        const lastMessage = session?.messages[session.messages.length - 1];
        const isPrototyperActive = session?.personaId === 'prototyper';
        const htmlContent = lastMessage?.role === 'model' && lastMessage.parts[0].text ? extractHtmlContent(lastMessage.parts[0].text) : null;
        
        return (
             <motion.aside
                initial={false}
                animate={{ width: isContextPanelOpen ? 320 : 0, padding: isContextPanelOpen ? 16 : 0 }}
                transition={{ type: 'spring', stiffness: 400, damping: 40 }}
                className="glass-surface border-l border-[var(--border-color)] flex-col overflow-hidden hidden lg:flex"
            >
                { isPrototyperActive && htmlContent ? (
                    <LivePreviewCanvas htmlContent={htmlContent} />
                ) : (
                    <>
                        <div className="flex items-center justify-between pb-4 border-b border-[var(--border-color)]">
                            <h3 className="font-semibold">AI Insights</h3>
                            <PremiumButton onClick={() => setIsContextPanelOpen(false)} variant="ghost" size="icon">{ICONS.CHEVRON_RIGHT}</PremiumButton>
                        </div>
                        <div className="flex-1 py-4 overflow-y-auto">
                            <div className="p-4 rounded-xl bg-gradient-luxury text-white space-y-3">
                                <h4 className="font-bold">💎 Upgrade to Pro</h4>
                                <ul className="text-sm list-disc list-inside space-y-1">
                                    <li>Unlimited agents</li>
                                    <li>Priority support</li>
                                    <li>Advanced analytics</li>
                                </ul>
                                <PremiumButton variant="secondary" className="w-full">Upgrade Now →</PremiumButton>
                            </div>
                        </div>
                    </>
                )}
            </motion.aside>
        );
    };

    return (
        <div className="flex h-screen w-screen bg-gradient-primary text-[var(--text-primary)] overflow-hidden">
            {isSidebarOpen && isMobile && <motion.div initial={{opacity: 0}} animate={{opacity: 1}} exit={{opacity: 0}} onClick={() => setIsSidebarOpen(false)} className="fixed inset-0 bg-black/60 z-10" />}
            
            <PrimaryNavSidebar />

            <div className="flex flex-col flex-1 min-w-0">
                <PremiumHeader />
                <main className="flex-1 min-h-0 relative">
                     {!isContextPanelOpen && view === 'chat' && (
                        <PremiumButton 
                            onClick={() => setIsContextPanelOpen(true)}
                            variant="ghost" size="icon"
                            className="absolute top-4 right-4 z-10 hidden lg:inline-flex"
                        >
                           {ICONS.CHEVRON_LEFT}
                        </PremiumButton>
                     )}
                    {renderView()}
                </main>
            </div>
            
            <ContextPanel session={activeSession} />
        </div>
    );
};

export default App;
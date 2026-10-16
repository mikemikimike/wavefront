import floConsoleService from '@app/api';
import { Alert, AlertDescription, AlertTitle } from '@app/components/ui/alert';
import { Button } from '@app/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@app/components/ui/dialog';
import { appEnv } from '@app/config/env';
import { TOKEN_KEY } from '@app/lib/constants';
import { extractErrorMessage } from '@app/lib/utils';
import { useNotifyStore } from '@app/store';
import { VoiceAgent, VoiceAgentRoom } from '@app/types/voice-agent';
import { PipecatClient } from '@pipecat-ai/client-js';
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';
import { AlertTriangle, Mic, MicOff, PhoneOff } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { BrowserCallError, classifyBrowserCallError, describeBrowserCallError } from './browser-call-errors';

type CallStatus = 'idle' | 'connecting' | 'connected' | 'ended' | 'error';

// The server gives up on a join that never completes after 60s; surface the failure a little sooner.
const CONNECT_TIMEOUT_MS = 45_000;
const ICE_SERVERS: RTCIceServer[] = [{ urls: 'stun:stun.l.google.com:19302' }];

interface BrowserCallDialogProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  agent: VoiceAgent;
  appId: string;
}

/**
 * Where the browser should POST its SDP offer. When floware proxies signaling the URL is relative to the
 * console gateway and needs the user's bearer token; otherwise it is the public call_processing URL and the
 * one-time join token is the credential.
 */
const buildSignalingRequest = (room: VoiceAgentRoom, appId: string) => {
  const base = room.signaling_url.replace(/\/+$/, '');
  const viaFloware = !!room.signaling_via_floware;
  const endpoint = viaFloware ? `${appEnv.baseURL}/v1/${appId}${base}/offer` : `${base}/offer`;
  const headers = new Headers();
  if (viaFloware) {
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) headers.set('Authorization', `Bearer ${token}`);
  }
  return {
    endpoint,
    headers,
    requestData: {
      room_id: room.room_id,
      voice_agent_id: room.voice_agent_id,
      join_token: room.join_token,
    },
  };
};

const withTimeout = <T,>(promise: Promise<T>, ms: number): Promise<T> =>
  new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Timed out waiting for the voice connection')), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });

const BrowserCallDialog: React.FC<BrowserCallDialogProps> = ({ isOpen, onOpenChange, agent, appId }) => {
  const { notifyError } = useNotifyStore();
  const [status, setStatus] = useState<CallStatus>('idle');
  const [error, setError] = useState<BrowserCallError | null>(null);
  const [micEnabled, setMicEnabled] = useState(true);

  const clientRef = useRef<PipecatClient | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Bumped on every start/teardown so late results from an abandoned attempt are ignored.
  const attemptRef = useRef(0);
  // True only while a call is established, so late transport events can't clobber connecting/error states.
  const liveRef = useRef(false);

  const teardown = async () => {
    attemptRef.current += 1;
    liveRef.current = false;
    const client = clientRef.current;
    clientRef.current = null;
    if (audioRef.current) audioRef.current.srcObject = null;
    if (client) {
      try {
        // Closes the peer connection and releases the microphone.
        await client.disconnect();
      } catch {
        // Already disconnected.
      }
    }
  };

  useEffect(() => {
    return () => {
      void teardown();
    };
  }, []);

  const fail = (attempt: number, callError: BrowserCallError, toast = true) => {
    if (attempt !== attemptRef.current) return;
    setError(callError);
    setStatus('error');
    if (toast) notifyError(callError.title);
  };

  const handleConnect = async () => {
    await teardown();
    const attempt = attemptRef.current;
    setError(null);
    setMicEnabled(true);
    setStatus('connecting');

    let room: VoiceAgentRoom | undefined;
    try {
      const response = await floConsoleService.voiceAgentService.createRoom(agent.id);
      room = response.data?.data;
    } catch (err) {
      // The shared axios instance already toasts API failures.
      fail(attempt, describeBrowserCallError('create_room_failed', extractErrorMessage(err)), false);
      return;
    }
    if (attempt !== attemptRef.current) return;
    if (!room) {
      fail(attempt, describeBrowserCallError('create_room_failed'));
      return;
    }

    const client = new PipecatClient({
      transport: new SmallWebRTCTransport({
        iceServers: ICE_SERVERS,
        // Gather candidates before the offer so the call does not depend on trickle-ICE PATCHes
        // reaching the same call_processing replica.
        waitForICEGathering: true,
      }),
      enableMic: true,
      enableCam: false,
      callbacks: {
        onTrackStarted: (track, participant) => {
          if (participant?.local || track.kind !== 'audio' || !audioRef.current) return;
          audioRef.current.srcObject = new MediaStream([track]);
          void audioRef.current.play().catch(() => undefined);
        },
        onDisconnected: () => {
          if (attempt !== attemptRef.current) return;
          // Server-side idle / max-duration limits end the session; so does the user hanging up
          // (which bumps attemptRef first and is ignored above).
          if (liveRef.current) setStatus('ended');
        },
        onTransportStateChanged: (state) => {
          if (attempt !== attemptRef.current || state !== 'error') return;
          if (!liveRef.current) return;
          setError(describeBrowserCallError('connection_failed'));
          setStatus('error');
        },
      },
    });
    clientRef.current = client;

    try {
      await withTimeout(
        client.connect({ webrtcRequestParams: buildSignalingRequest(room, appId) }),
        CONNECT_TIMEOUT_MS
      );
      if (attempt !== attemptRef.current) return;
      liveRef.current = true;
      setStatus('connected');
    } catch (err) {
      const callError = await classifyBrowserCallError(err);
      if (attempt !== attemptRef.current) return;
      await teardown();
      // teardown() invalidated this attempt; re-assert the error state directly.
      setError(callError);
      setStatus('error');
      notifyError(callError.title);
    }
  };

  const handleHangUp = async () => {
    await teardown();
    setStatus('idle');
  };

  const handleToggleMic = () => {
    const next = !micEnabled;
    clientRef.current?.enableMic(next);
    setMicEnabled(next);
  };

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      void teardown();
      setStatus('idle');
      setError(null);
    }
    onOpenChange(open);
  };

  const isBusy = status === 'connecting';
  const isLive = status === 'connected';

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Talk in browser</DialogTitle>
          <DialogDescription>
            Speak with <strong>{agent.name}</strong> using your microphone. No phone number is needed.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4" aria-live="polite">
          {status === 'idle' && (
            <p className="frost-text-muted text-sm">
              Your browser will ask for microphone access when you connect. Each session uses a new one-time room.
            </p>
          )}

          {isBusy && <p className="frost-text-muted text-sm">Connecting to {agent.name}...</p>}

          {isLive && (
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-medium">Connected. Start speaking.</p>
              <Button type="button" variant="outline" size="sm" onClick={handleToggleMic}>
                {micEnabled ? <Mic /> : <MicOff />}
                {micEnabled ? 'Mute' : 'Unmute'}
              </Button>
            </div>
          )}

          {status === 'ended' && (
            <Alert variant="info">
              <AlertTitle>Call ended</AlertTitle>
              <AlertDescription>
                The session finished (idle or time limit reached). Start a new session to keep talking.
              </AlertDescription>
            </Alert>
          )}

          {status === 'error' && error && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>{error.title}</AlertTitle>
              <AlertDescription>{error.message}</AlertDescription>
            </Alert>
          )}

          {/* Plays the agent's voice; the transport does not attach remote audio itself. */}
          <audio ref={audioRef} autoPlay />
        </div>

        <DialogFooter>
          {isLive ? (
            <Button type="button" variant="destructive" onClick={handleHangUp}>
              <PhoneOff />
              Hang up
            </Button>
          ) : (
            <>
              <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
                Close
              </Button>
              <Button type="button" onClick={handleConnect} disabled={isBusy} loading={isBusy}>
                {status === 'error' ? 'Retry' : status === 'ended' ? 'Start new session' : 'Connect'}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default BrowserCallDialog;

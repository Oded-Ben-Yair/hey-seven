/** Message roles in the chat */
export type MessageRole = "user" | "assistant";

/** Specialized message content types beyond plain text */
export type MessageType =
  | "text"
  | "comp_offer"
  | "reservation_confirmation"
  | "escalation_notice";

/** A single chat message */
export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  type: MessageType;
  timestamp: Date;
  metadata?: CompOfferMetadata | ReservationMetadata | EscalationMetadata;
}

/** Metadata attached to comp offer messages */
export interface CompOfferMetadata {
  compType: string;
  value: number;
  currency: string;
  expiresAt: string;
}

/** Metadata attached to reservation confirmation messages */
export interface ReservationMetadata {
  venue: string;
  date: string;
  time: string;
  partySize: number;
  confirmationId: string;
}

/** Metadata attached to escalation notices */
export interface EscalationMetadata {
  reason: string;
  assignedTo: string;
  priority: "low" | "medium" | "high" | "urgent";
}

/** Player tier levels */
export type PlayerTier = "Gold" | "Platinum" | "Diamond" | "Seven Star";

/** Player profile data */
export interface PlayerProfile {
  id: string;
  name: string;
  tier: PlayerTier;
  photoUrl: string | null;
  adt: number;
  compBalance: number;
  lastVisit: string;
  totalVisits: number;
  memberSince: string;
  preferences: string[];
  recentActivity: VisitSummary[];
}

/** Summary of a player visit */
export interface VisitSummary {
  date: string;
  duration: string;
  spend: number;
  primaryGame: string;
}

/** Response from the chat API */
export interface ChatResponse {
  message: ChatMessage;
  playerContext?: Partial<PlayerProfile>;
}

/** Response from the comp calculation API */
export interface CompResult {
  compType: string;
  calculatedValue: number;
  maxAllowed: number;
  approved: boolean;
  reason: string;
}

/** Connection status for the header indicator */
export type ConnectionStatus = "online" | "processing" | "offline";

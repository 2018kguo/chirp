//! Who is calling, and how: what a brain factory sees before a call starts,
//! to choose the call's brain (a user's chat, a voice) or refuse it.

use std::collections::HashMap;
use std::net::SocketAddr;

use axum::http::HeaderMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    WebSocket,
    WebRtc,
    Sip,
}

#[derive(Clone, Debug)]
pub struct CallRequest {
    pub transport: Transport,
    /// The caller's address (for SIP, where its INVITE came from), when known.
    pub remote: Option<SocketAddr>,
    /// The HTTP request's headers (WebSocket, WebRTC); empty for SIP.
    pub headers: HeaderMap,
    /// The call URL's query parameters (WebSocket, WebRTC); for SIP, `from`,
    /// `to` and `call_id` from the INVITE.
    pub query: HashMap<String, String>,
}

impl CallRequest {
    /// The caller's token: `Authorization: Bearer <token>`, else `?token=`
    /// (a browser can't set headers on a WebSocket).
    pub fn token(&self) -> Option<&str> {
        self.headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .or_else(|| self.query.get("token").map(String::as_str))
    }

    /// Whether the caller presented `expected` (compared in constant time).
    pub fn has_token(&self, expected: &str) -> bool {
        let Some(token) = self.token() else {
            return false;
        };
        token.len() == expected.len()
            && token
                .bytes()
                .zip(expected.bytes())
                .fold(0u8, |difference, (a, b)| difference | (a ^ b))
                == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(header: Option<&str>, query: Option<&str>) -> CallRequest {
        let mut headers = HeaderMap::new();
        if let Some(header) = header {
            headers.insert(axum::http::header::AUTHORIZATION, header.parse().unwrap());
        }
        CallRequest {
            transport: Transport::WebSocket,
            remote: None,
            headers,
            query: query
                .map(|token| ("token".to_string(), token.to_string()))
                .into_iter()
                .collect(),
        }
    }

    #[test]
    fn a_token_comes_from_the_header_or_the_query() {
        assert!(request(Some("Bearer s3cret"), None).has_token("s3cret"));
        assert!(request(None, Some("s3cret")).has_token("s3cret"));
        assert!(!request(Some("Bearer s3cre"), None).has_token("s3cret"));
        assert!(!request(Some("Basic s3cret"), None).has_token("s3cret"));
        assert!(!request(None, None).has_token("s3cret"));
    }
}

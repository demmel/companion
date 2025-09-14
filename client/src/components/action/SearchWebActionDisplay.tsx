import { useState } from "react";
import { css } from "@styled-system/css";
import { Loader2, Search, ChevronDown, ExternalLink } from "lucide-react";
import { SearchWebAction } from "@/types";

interface SearchWebActionDisplayProps {
  action: SearchWebAction;
}

export function SearchWebActionDisplay({
  action,
}: SearchWebActionDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const isStreaming = action.status.type === "streaming";
  const result =
    action.status.type === "error"
      ? `Error: ${action.status.error}`
      : action.status.result;
  const hasContent =
    result?.trim().length > 0 ||
    (action.search_results && action.search_results.length > 0);

  return (
    <div
      className={css({
        display: "flex",
        flexDirection: "column",
        gap: 2,
        p: 3,
        bg: "blue.900/20",
        border: "1px solid",
        borderColor: "blue.700",
        rounded: "md",
        fontSize: "sm",
      })}
    >
      {/* Header with icon and search query */}
      <div className={css({ display: "flex", alignItems: "center", gap: 2 })}>
        {isStreaming ? (
          <Loader2
            size={16}
            className={css({
              animation: "spin 1s linear infinite",
              color: "blue.500",
            })}
          />
        ) : (
          <Search size={16} className={css({ color: "blue.500" })} />
        )}

        <div className={css({ flex: 1 })}>
          <div className={css({ color: "blue.300", fontWeight: "medium" })}>
            {isStreaming
              ? "Searching web..."
              : action.status.type === "error"
                ? "Search failed"
                : `Found ${action.search_results?.length || 0} results`}
          </div>
          {action.query && (
            <div
              className={css({
                color: "blue.200",
                fontSize: "xs",
                truncate: true,
              })}
            >
              "{action.query}"
            </div>
          )}
        </div>
      </div>

      {/* Purpose context */}
      {action.purpose && (
        <div
          className={css({
            color: "blue.400",
            fontSize: "xs",
            fontStyle: "italic",
          })}
        >
          Purpose: {action.purpose}
        </div>
      )}

      {/* Expandable result content */}
      {(hasContent || isStreaming) && (
        <div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className={css({
              w: "full",
              px: 2,
              py: 1,
              textAlign: "left",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              fontSize: "sm",
              color: "blue.400",
              _hover: {
                bg: "blue.800/30",
              },
              cursor: "pointer",
            })}
          >
            <span>{isStreaming ? "Searching..." : "Search Results"}</span>
            <ChevronDown
              size={14}
              className={css({
                transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
                transition: "transform 0.2s ease",
              })}
            />
          </button>

          {isExpanded && (
            <div
              className={css({
                color: "blue.100",
                fontSize: "sm",
                lineHeight: "relaxed",
                p: 2,
                bg: "blue.950/50",
                rounded: "sm",
                border: "1px solid",
                borderColor: "blue.800",
                mx: 2,
                mb: 2,
              })}
            >
              {action.status.type === "error" ? (
                result?.trim() || "Search failed"
              ) : isStreaming ? (
                <div>
                  Searching for results...
                  <span
                    className={css({
                      animation: "blink 1s infinite",
                      color: "blue.500",
                    })}
                  >
                    ▍
                  </span>
                </div>
              ) : action.search_results && action.search_results.length > 0 ? (
                <div
                  className={css({
                    display: "flex",
                    flexDirection: "column",
                    gap: 3,
                  })}
                >
                  {action.search_results.map((searchResult, index) => (
                    <div
                      key={index}
                      className={css({
                        p: 2,
                        bg: "blue.900/30",
                        border: "1px solid",
                        borderColor: "blue.800",
                        rounded: "sm",
                      })}
                    >
                      <div
                        className={css({
                          display: "flex",
                          alignItems: "flex-start",
                          gap: 2,
                          mb: 1,
                        })}
                      >
                        <ExternalLink
                          size={12}
                          className={css({
                            color: "blue.500",
                            mt: 0.5,
                            flexShrink: 0,
                          })}
                        />
                        <div className={css({ flex: 1, minWidth: 0 })}>
                          <a
                            href={searchResult.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className={css({
                              color: "blue.200",
                              fontWeight: "medium",
                              textDecoration: "underline",
                              _hover: { color: "blue.100" },
                              fontSize: "sm",
                              lineHeight: "tight",
                              display: "block",
                            })}
                          >
                            {searchResult.title}
                          </a>
                          <div
                            className={css({
                              color: "blue.400",
                              fontSize: "xs",
                              mt: 1,
                              truncate: true,
                            })}
                          >
                            {searchResult.url}
                          </div>
                        </div>
                      </div>
                      {searchResult.snippet && (
                        <div
                          className={css({
                            color: "blue.100",
                            fontSize: "xs",
                            lineHeight: "relaxed",
                            mt: 2,
                            pl: 4,
                          })}
                        >
                          {searchResult.snippet}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                "No results found"
              )}
            </div>
          )}
        </div>
      )}

      <style>
        {`
          @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
          }
        `}
      </style>
    </div>
  );
}

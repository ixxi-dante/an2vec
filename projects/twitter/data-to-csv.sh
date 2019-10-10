#!/bin/bash -e
# Create a mutual-mention network and a list of tweets per user from a sosweet dataset

if [ $# -eq 0 ]; then
  echo "Usage: $(basename $0) FILE [FILE ...]"
fi

function mention_network() {
  echo "Extracting mutual-mention network from $1"
  zcat $1 \
    | sed 's/^[^\{]*//g' \
    | jq -R 'fromjson?' \
    | jq -c '. | [[.actor.id | ltrimstr("id:twitter.com:") | tonumber ], [.twitter_entities.user_mentions[].id]] | combinations' \
    | sed 's/\[\|\]//g' \
    | sort \
    | uniq -c \
    | sed 's/^ \+//g' \
    | sed 's/ /,/g' \
    > "$2"
}

function user_tweets() {
  echo "Extracting user tweets from $1"
  zcat $1 \
    | sed 's/^[^\{]*//g' \
    | jq -R 'fromjson?' \
    | jq -c '. | [(.actor.id | ltrimstr("id:twitter.com:") | tonumber),.body]' \
    | sed 's/\[\|\]//g' \
    | sed 's/,/ /g' \
    | sed 's!\S*\.\S*!!g' \
    | sed 's/\"//g' \
    | sed 's/@\S* //g' \
    | sed 's/\\n//g' \
    | iconv -f utf-8 -t ascii//translit \
    | sed 's/#/hashtagreplace/g' \
    | tr '[:punct:]' ' ' \
    | sed 's/hashtagreplace/#/g' \
    | tr "[:upper:]" "[:lower:]" \
    | tr -s " " \
    | sort \
    > "$2"
}

function check_file_absent() {
  if [ -f "$1" ]; then
    echo "File '$1' already exists, not overwriting it."
    echo "Aborting."
    exit 1
  fi
}

for FILE in "$@"; do
  BASE=${FILE%.*}

  NETWORK="${BASE}-mutual_mention_network.csv"
  check_file_absent "$NETWORK"
  mention_network $FILE $NETWORK

  TWEETS="${BASE}-user_tweets.csv"
  check_file_absent $TWEETS
  user_tweets $FILE $TWEETS
done

#!/bin/bash -e
# Create a mutual-mention network and a list of tweets per user from a sosweet dataset

if [ $# -lt 2 ]; then
  echo "Usage: $(basename $0) OUTNAME FILE [FILE ...]"
fi

function mention_network() {
  OUT=$1
  shift
  echo "Extracting mutual-mention network from $@"
  zcat $@ \
    | sed 's/^[^\{]*//g' \
    | jq -R 'fromjson?' \
    | jq -c '. | [[.actor.id | ltrimstr("id:twitter.com:") | tonumber], [.twitter_entities.user_mentions[].id]] | combinations' \
    | sed 's/\[\|\]//g' \
    | sort \
    | uniq -c \
    | sed 's/^ \+//g' \
    | sed 's/ /,/g' \
    > "$OUT"
}

function user_tweets() {
  OUT=$1
  shift
  echo "Extracting user tweets from $@"
  zcat $@ \
    | sed 's/^[^\{]*//g' \
    | jq -R 'fromjson?' \
    | jq -c '. | [(.actor.id | ltrimstr("id:twitter.com:") | tonumber), .body]' \
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
    > "$OUT"
}

function check_file_absent() {
  if [ -f "$1" ]; then
    echo "File '$1' already exists, not overwriting it."
    echo "Aborting."
    exit 1
  fi
}

OUTNAME=$1
shift

NETWORK="${OUTNAME}-mutual_mention_network.csv"
check_file_absent $NETWORK
mention_network $NETWORK $@

TWEETS="${OUTNAME}-user_tweets.csv"
check_file_absent $TWEETS
user_tweets $TWEETS $@

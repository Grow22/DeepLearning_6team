# AI 이미지 생성 프롬프트 (파일럿 50장)

- 같은 프롬프트 25개를 **ChatGPT 25장, Gemini 25장**에 똑같이 넣는다. 생성기만 다르고 장면은 같아야 생성기끼리 비교할 수 있다.
- 한 대화창에서 연달아 만들면 스타일이 비슷해지므로 **5장마다 새 채팅**을 연다.
- 비율은 **가로 3:2**로 요청한다. 진짜 사진(위키미디어)이 대부분 가로 사진이라 비율 차이를 모델이 외우지 않게 하려는 것이다.
- 실존 인물(정치인), 정당 이름, 로고는 절대 넣지 않는다. 연설자는 모두 **가상 인물**이다.
- 장면이 서양처럼 나오면(영어 간판, 서양인) 버리고 다시 만든다. 진짜 사진은 전부 한국 장면이라, AI 쪽만 서양 장면이면 모델이 "서양 풍경 = AI"를 외운다.
- 받은 파일은 **변환하지 말고 그대로 저장**한다. 포맷 통일은 나중에 코드로 한꺼번에 한다.
- 보이는 워터마크(예: Gemini 오른쪽 아래 반짝이 아이콘)는 지우지 말고 meta.csv의 `note` 칸에 적어둔다. 나중에 코드로 일괄 크롭한다.

## 파일 이름

`ai_{generator}_{category}_{번호}.png`
예: `ai_chatgpt_crowd_01.png`, `ai_gemini_vote_03.png`
번호는 아래 프롬프트 번호(카테고리 안의 01~05)를 그대로 쓴다.

## 공통 앞부분 (매번 붙이기)

```
Generate a photorealistic image, 3:2 landscape. It should look like an ordinary
photo taken by a person on a smartphone or a news camera, not an illustration.
Natural lighting, slight imperfections, no text overlay, no frame.
The location is South Korea and all people are Korean; any signs are in Korean.
Use only fictional party names and fictional candidate names. Do not show any real Korean political party names or logos.
Scene:
```

## 1. crowd (유세·집회 군중)

1. A large crowd gathered at a city square in Seoul for an election rally, people holding small signs and flags, daytime, seen from slightly above.
2. An evening candlelight rally on a wide Seoul boulevard, thousands of people holding candles and paper cups, buildings lit in the background.
3. Supporters in matching colored vests cheering at a campaign event in front of a subway station exit, mid-afternoon, crowded sidewalk.
4. A rally seen from behind the crowd, backs of people's heads, a distant stage with a large screen, overcast sky.
5. A protest march moving down a Korean city street, people carrying banners, police officers in reflective vests along the side.

## 2. street (거리·현수막)

1. A Korean street with election campaign banners hung between lamp posts, cars and pedestrians, overcast day.
2. Election candidate posters attached to a long wall on a residential street in Korea, an elderly person walking past.
3. A busy intersection in a Korean city with a large vertical campaign banner on a building facade, crosswalk in the foreground.
4. A campaign truck with loudspeakers parked on the side of a Korean road, a few staff members standing nearby, daytime.
5. A narrow alley in an older Korean neighborhood with a small election banner hanging over a shop, afternoon light.

## 3. speech (연설·행사장, 가상 인물)

1. A middle-aged male politician (fictional person) giving a speech at an outdoor podium with microphones, campaign banner behind him, seen from the audience.
2. A female candidate in her fifties (fictional person) speaking on a small truck-mounted stage, holding a microphone, supporters in front.
3. A press conference in a Korean government briefing room, a speaker (fictional person) at a lectern, rows of reporters with cameras in the foreground.
4. An indoor campaign event in a hall, an empty stage with a podium and a large backdrop, staff arranging chairs.
5. A candidate (fictional person) shaking hands with citizens at a traditional market, seen from a few meters away.

## 4. vote (투표소)

1. Inside a Korean polling station in a school gymnasium, voters waiting in line, a ballot box on a table, indoor fluorescent lighting.
2. Close view of a transparent ballot box on a table with election staff sitting behind it, polling station interior.
3. Voting booths with curtains lined up in a community center, one voter stepping into a booth.
4. A line of people waiting outside a polling station entrance with a sign on the door, morning light.
5. Election staff counting ballots at long tables in a large hall, many people working, overhead view.

## 5. city (일반 도시 풍경)

1. Seoul City Hall and the plaza in front of it, people walking, clear daytime.
2. A wide plaza in central Seoul with a statue and pedestrians, mountains in the background.
3. A busy downtown street in Seoul at dusk with traffic and shop signs.
4. A riverside park in a Korean city on a weekend, people walking and cycling, apartment buildings across the river.
5. A subway station exit on a Korean city street during rush hour, people coming out, bus stop nearby.
